from __future__ import annotations

import numpy as np

from .core import generalized_eigh
from .params import CCIKThickRestartParams
from .propagation import build_next_krylov_vector, format_krylov_step_diagnostics, krylov_step_diagnostics_as_dict
from .selection import build_projection_workspace, candidate_mode_from_params, compression_policy_from_params, make_full_hdiag_map
from .sparse_ci import (
    SparseCIVector,
    compress_topk_sparse,
    inner_sparse,
    linear_combination_sparse,
    make_hf_sparse_vector,
    normalize_sparse,
    orthogonalize_sparse,
    sparse_norm,
    threshold_prune_sparse,
)



def gram_schmidt_list(vecs: list[SparseCIVector], tol: float = 1e-12) -> list[SparseCIVector]:
    """Orthonormalize list of sparse CI vectors; drop near-dependent ones."""
    Q: list[SparseCIVector] = []
    for v in vecs:
        vn = normalize_sparse(v)
        if not vn.coeffs:
            continue
        r = orthogonalize_sparse(vn, Q)
        nr = sparse_norm(r)
        if nr > tol:
            Q.append(normalize_sparse(r))
    return Q


def _compress_with_stage1(vec: SparseCIVector, nkeep: int, params: CCIKThickRestartParams) -> SparseCIVector:
    """Use an oversampled first truncation before the final keep size."""
    compression_policy = compression_policy_from_params(params, default="topk")
    if compression_policy == "threshold":
        unit = normalize_sparse(vec)
        out, _, _ = threshold_prune_sparse(unit, float(getattr(params, "store_abs_tol", 0.0)), keep_largest_if_empty=True)
        return out
    if compression_policy == "fixed_budget":
        return compress_topk_sparse(vec, nkeep=max(1, int(getattr(params, "fixed_store_keep", 1_000_000))))
    nraw = len(vec.coeffs)
    nkeep_stage1 = max(nkeep, min(nraw, 8 * nkeep))
    trial = compress_topk_sparse(vec, nkeep=nkeep_stage1)
    return compress_topk_sparse(trial, nkeep=nkeep)


def _cycle_ritz(
    *,
    Q: list[SparseCIVector],
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: CCIKThickRestartParams,
    hdiag_full: dict[tuple[int, int], float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    m_eff = len(Q)
    Hproj = np.zeros((m_eff, m_eff))
    Sproj = np.zeros((m_eff, m_eff))
    Hq = [
        build_projection_workspace(
            qk=Q[j],
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        ).v_proj
        for j in range(m_eff)
    ]
    for i in range(m_eff):
        for j in range(m_eff):
            Sproj[i, j] = inner_sparse(Q[i], Q[j])
            Hproj[i, j] = inner_sparse(Q[i], Hq[j])
    return generalized_eigh(Hproj, Sproj)



def _ccik_cycle_sparse_multi_start(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    start_vecs: list[SparseCIVector],
    params: CCIKThickRestartParams,
    hdiag_full: dict[tuple[int, int], float] | None,
) -> tuple[np.ndarray, np.ndarray, list[SparseCIVector]]:
    """One thick-restart cycle: build a compressed sparse Krylov basis from multiple starts."""

    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    Q = gram_schmidt_list(start_vecs, tol=float(getattr(params, "orth_tol", 1e-12)))
    if len(Q) == 0:
        raise RuntimeError("All start vectors vanished after orthogonalization.")

    frontier = list(range(len(Q)))
    head = 0

    while len(Q) < params.m_cycle and head < len(frontier):
        parent_idx = frontier[head]
        head += 1
        qk = Q[parent_idx]
        q_next, diag, _ = build_next_krylov_vector(
            qk,
            Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        if params.verbose:
            print(format_krylov_step_diagnostics(f"      |Q|={len(Q):02d} parent={parent_idx+1:02d}", diag))
        if not q_next.coeffs:
            break
        Q.append(q_next)
        frontier.append(len(Q) - 1)

    evals, X = _cycle_ritz(
        Q=Q,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )
    return evals, X, Q



def ccik_ground_energy_dense_thick_restart(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKThickRestartParams | None = None,
    stats: dict[str, object] | None = None,
) -> float:
    """Sparse selected-space CCIK with thick restart."""

    if params is None:
        params = CCIKThickRestartParams()

    checkpoint_every: int | None = None
    total_max: int | None = None
    if (
        int(params.ncycles) > 0
        and int(params.m_cycle) > int(params.ncycles)
        and int(params.m_cycle) % int(params.ncycles) == 0
    ):
        checkpoint_every = int(params.ncycles)
        total_max = int(params.m_cycle)

    from pyscf.fci import direct_spin1

    mode = candidate_mode_from_params(params, default="selected_ci")
    hdiag_full: dict[tuple[int, int], float] | None = None
    if mode == "full":
        hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=norb, nelec=nelec)

    if checkpoint_every is not None and total_max is not None:
        Q: list[SparseCIVector] = [q0]
        prev_E: float | None = None
        ncheck_done = 0
        if stats is not None:
            stats["step_diagnostics"] = []

        while len(Q) < total_max:
            qk = Q[-1]
            q_next, diag, _ = build_next_krylov_vector(
                qk,
                Q,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
            if params.verbose:
                print(format_krylov_step_diagnostics(f"    checkpoint m={len(Q):02d}", diag))
            if stats is not None:
                step_diags = stats.setdefault("step_diagnostics", [])
                if isinstance(step_diags, list):
                    step_diags.append(krylov_step_diagnostics_as_dict(diag))
                stats["last_step_diagnostics"] = krylov_step_diagnostics_as_dict(diag)
            if not q_next.coeffs:
                break
            Q.append(q_next)

            if (len(Q) % checkpoint_every) == 0:
                evals, _ = _cycle_ritz(
                    Q=Q,
                    h1=h1,
                    eri8=eri8,
                    h2eff=h2eff,
                    norb=norb,
                    nelec=nelec,
                    params=params,
                    hdiag_full=hdiag_full,
                )
                E0 = float(evals[0])
                ncheck_done += 1

                if params.verbose:
                    if prev_E is None:
                        print(f"    [CCIK checkpoint 1] m={len(Q)} E0={E0:+.10f}")
                    else:
                        print(f"    [CCIK checkpoint {ncheck_done}] m={len(Q)} E0={E0:+.10f} Δ={E0-prev_E:+.3e}")

                if prev_E is not None and abs(E0 - prev_E) < float(params.tol):
                    if stats is not None:
                        union = set()
                        ndet_sum = 0
                        for q in Q:
                            supp = q.support()
                            union.update(supp)
                            ndet_sum += len(supp)
                        stats["cycle"] = int(ncheck_done)
                        stats["m_eff"] = int(len(Q))
                        stats["ndet_union"] = int(len(union))
                        stats["ndet_sum"] = int(ndet_sum)
                        stats["checkpoint_every"] = int(checkpoint_every)
                    return E0

                prev_E = E0

        evals, _ = _cycle_ritz(
            Q=Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        if stats is not None:
            union = set()
            ndet_sum = 0
            for q in Q:
                supp = q.support()
                union.update(supp)
                ndet_sum += len(supp)
            stats["cycle"] = int(ncheck_done)
            stats["m_eff"] = int(len(Q))
            stats["ndet_union"] = int(len(union))
            stats["ndet_sum"] = int(ndet_sum)
            stats["checkpoint_every"] = int(checkpoint_every)
        return float(evals[0])

    starts: list[SparseCIVector] = [q0]
    E_prev = None

    for cyc in range(params.ncycles):
        evals, X, Q = _ccik_cycle_sparse_multi_start(
            h1,
            eri8,
            norb,
            nelec,
            start_vecs=starts,
            params=params,
            hdiag_full=hdiag_full,
        )

        nroot_eff = min(params.nroot, X.shape[1])
        new_starts: list[SparseCIVector] = []
        for r in range(nroot_eff):
            coeffs = X[:, r]
            psi = linear_combination_sparse((float(coeffs[i]), Q[i]) for i in range(len(Q)))
            psi = normalize_sparse(_compress_with_stage1(psi, int(params.nkeep), params))
            if psi.coeffs:
                new_starts.append(psi)

        starts = gram_schmidt_list(new_starts, tol=float(getattr(params, "orth_tol", 1e-12)))
        if len(starts) == 0 and len(new_starts) > 0:
            starts = [new_starts[0]]

        E0 = float(evals[0])

        if stats is not None:
            union = set()
            ndet_sum = 0
            for Qi in Q:
                supp = Qi.support()
                union.update(supp)
                ndet_sum += len(supp)
            stats["cycle"] = int(cyc + 1)
            stats["m_eff"] = int(len(Q))
            stats["ndet_union"] = int(len(union))
            stats["ndet_sum"] = int(ndet_sum)
            stats["nstart"] = int(len(starts))

        if params.verbose:
            if E_prev is None:
                print(f"    [CCIK cycle {cyc+1}] E0={E0:+.10f}")
            else:
                print(f"    [CCIK cycle {cyc+1}] E0={E0:+.10f} Δ={E0-E_prev:+.3e}")

        if E_prev is not None and abs(E0 - E_prev) < params.tol:
            return E0
        E_prev = E0

    if E_prev is None:
        raise RuntimeError("CCIK thick restart did not produce an energy")

    return float(E_prev)
