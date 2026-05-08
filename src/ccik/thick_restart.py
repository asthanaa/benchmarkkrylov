"""Thick-restart CCIK, the recommended default solver in this repository."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .params import CCIKThickRestartParams
from .propagation import build_next_krylov_vector, format_krylov_step_diagnostics, krylov_step_diagnostics_as_dict
from .ritz import projected_ritz_mode_from_params, solve_projected_ritz
from .selection import (
    _enumerate_connected_sd_support,
    candidate_mode_from_params,
    compression_policy_from_params,
    make_full_hdiag_map,
)
from .sparse_ci import (
    DetKey,
    SparseCIVector,
    compress_topk_sparse,
    compress_topk_sparse_grouped,
    linear_combination_sparse,
    make_hf_sparse_vector,
    normalize_sparse,
    orthogonalize_sparse,
    sparse_from_scivector,
    sparse_norm,
    threshold_prune_sparse,
)


@dataclass(frozen=True)
class ThickRestartIterationRecord:
    """Record one thick-restart Krylov expansion checkpoint."""

    cycle: int
    basis_size: int
    restart_seed_count: int
    cum_new_krylov: int
    electronic_energy_eh: float
    support_union_size: int | None = None
    exactized_electronic_energy_eh: float | None = None
    exactized_solver_kind: str | None = None
    exactized_subspace_dim: int | None = None
    exactized_hamiltonian_nnz: int | None = None
    exactizeplus_electronic_energy_eh: float | None = None
    exactizeplus_solver_kind: str | None = None
    exactizeplus_subspace_dim: int | None = None
    exactizeplus_parent_count: int | None = None
    exactizeplus_shell_added: int | None = None


def _maybe_trace_iteration(
    iteration_callback: Callable[[ThickRestartIterationRecord], None] | None,
    *,
    cycle: int,
    basis_size: int,
    restart_seed_count: int,
    cum_new_krylov: int,
    electronic_energy_eh: float,
    support_union_size: int | None = None,
    exactized_electronic_energy_eh: float | None = None,
    exactized_solver_kind: str | None = None,
    exactized_subspace_dim: int | None = None,
    exactized_hamiltonian_nnz: int | None = None,
    exactizeplus_electronic_energy_eh: float | None = None,
    exactizeplus_solver_kind: str | None = None,
    exactizeplus_subspace_dim: int | None = None,
    exactizeplus_parent_count: int | None = None,
    exactizeplus_shell_added: int | None = None,
) -> None:
    if iteration_callback is None:
        return
    iteration_callback(
        ThickRestartIterationRecord(
            cycle=int(cycle),
            basis_size=int(basis_size),
            restart_seed_count=int(restart_seed_count),
            cum_new_krylov=int(cum_new_krylov),
            electronic_energy_eh=float(electronic_energy_eh),
            support_union_size=None if support_union_size is None else int(support_union_size),
            exactized_electronic_energy_eh=(
                None if exactized_electronic_energy_eh is None else float(exactized_electronic_energy_eh)
            ),
            exactized_solver_kind=None if exactized_solver_kind is None else str(exactized_solver_kind),
            exactized_subspace_dim=None if exactized_subspace_dim is None else int(exactized_subspace_dim),
            exactized_hamiltonian_nnz=(
                None if exactized_hamiltonian_nnz is None else int(exactized_hamiltonian_nnz)
            ),
            exactizeplus_electronic_energy_eh=(
                None if exactizeplus_electronic_energy_eh is None else float(exactizeplus_electronic_energy_eh)
            ),
            exactizeplus_solver_kind=(
                None if exactizeplus_solver_kind is None else str(exactizeplus_solver_kind)
            ),
            exactizeplus_subspace_dim=(
                None if exactizeplus_subspace_dim is None else int(exactizeplus_subspace_dim)
            ),
            exactizeplus_parent_count=(
                None if exactizeplus_parent_count is None else int(exactizeplus_parent_count)
            ),
            exactizeplus_shell_added=(
                None if exactizeplus_shell_added is None else int(exactizeplus_shell_added)
            ),
        )
    )


def _support_union_keys(Q: list[SparseCIVector]) -> list[DetKey]:
    union: set[DetKey] = set()
    for qvec in Q:
        union.update((int(stra), int(strb)) for stra, strb in qvec.coeffs)
    return sorted(union, key=lambda key: (int(key[0]), int(key[1])))


def _checkpoint_period_from_params(params: CCIKThickRestartParams) -> int | None:
    raw = getattr(params, "exactized_checkpoint_every", None)
    if raw is None:
        return None
    period = int(raw)
    return None if period <= 0 else period


def _exactizeplus_enabled(params: CCIKThickRestartParams) -> bool:
    return bool(getattr(params, "exactizeplus_enabled", False))


def _exactizeplus_top_parents(params: CCIKThickRestartParams) -> int:
    return max(1, int(getattr(params, "exactizeplus_top_parents", 500)))


def _restricted_ground_state_guess(
    Q: list[SparseCIVector],
    projected_eigvecs: np.ndarray | None,
    support_keys: list[DetKey],
) -> np.ndarray | None:
    if projected_eigvecs is None or projected_eigvecs.size == 0 or len(Q) == 0 or len(support_keys) == 0:
        return None
    coeffs = np.asarray(projected_eigvecs[:, 0], dtype=float)
    psi = normalize_sparse(linear_combination_sparse((float(coeffs[idx]), Q[idx]) for idx in range(len(Q))))
    if not psi.coeffs:
        return None
    guess = np.asarray([float(psi.coeffs.get(key, 0.0)) for key in support_keys], dtype=float)
    nrm = float(np.linalg.norm(guess))
    if nrm <= 0.0:
        return None
    return guess / nrm


def _exactize_support_union(
    *,
    support_keys: list[DetKey],
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    initial_guess: np.ndarray | None,
) -> dict[str, object]:
    from pyscf.fci import selected_ci

    dim = int(len(support_keys))
    if dim <= 0:
        raise ValueError("Cannot exactize an empty determinant support")

    strsa = np.asarray(sorted({int(stra) for stra, _ in support_keys}), dtype=np.int64)
    strsb = np.asarray(sorted({int(strb) for _, strb in support_keys}), dtype=np.int64)
    map_a = {int(stra): idx for idx, stra in enumerate(strsa.tolist())}
    map_b = {int(strb): idx for idx, strb in enumerate(strsb.tolist())}
    positions = np.asarray(
        [(int(map_a[int(stra)]), int(map_b[int(strb)])) for stra, strb in support_keys],
        dtype=np.int64,
    )

    def _vector_to_scivector(coeffs: np.ndarray):
        arr = np.zeros((len(strsa), len(strsb)), dtype=float)
        nz = np.flatnonzero(np.abs(coeffs) > 0.0)
        if nz.size:
            arr[positions[nz, 0], positions[nz, 1]] = coeffs[nz]
        return selected_ci._as_SCIvector(arr, (strsa, strsb))

    def _project_scivector(ci_sci) -> np.ndarray:
        arr = np.asarray(ci_sci, dtype=float)
        return np.asarray(arr[positions[:, 0], positions[:, 1]], dtype=float)

    def _restricted_matvec(coeffs: np.ndarray) -> np.ndarray:
        sci_vec = _vector_to_scivector(np.asarray(coeffs, dtype=float))
        hq_sci = selected_ci.contract_2e(h2eff, sci_vec, norb, nelec)
        return _project_scivector(hq_sci)

    if dim == 1:
        energy = float(_restricted_matvec(np.asarray([1.0], dtype=float))[0])
        return {
            "electronic_energy_eh": energy,
            "solver_kind": "dense_eigh",
            "subspace_dim": dim,
            "hamiltonian_nnz": 1,
            "lowest_eigenvector": np.asarray([1.0], dtype=float),
        }

    if dim <= 512:
        H = np.zeros((dim, dim), dtype=float)
        for col in range(dim):
            basis_vec = np.zeros(dim, dtype=float)
            basis_vec[col] = 1.0
            H[:, col] = _restricted_matvec(basis_vec)
        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)
        return {
            "electronic_energy_eh": float(evals[0]),
            "solver_kind": "dense_eigh",
            "subspace_dim": dim,
            "hamiltonian_nnz": int(np.count_nonzero(np.abs(H) > 1.0e-14)),
            "lowest_eigenvector": np.asarray(evecs[:, 0], dtype=float),
        }

    from scipy.sparse.linalg import LinearOperator, eigsh

    def _matvec(coeffs: np.ndarray) -> np.ndarray:
        return _restricted_matvec(np.asarray(coeffs, dtype=float))

    operator = LinearOperator((dim, dim), matvec=_matvec, dtype=float)
    eigsh_kwargs: dict[str, object] = {
        "k": 1,
        "which": "SA",
        "tol": 1.0e-10,
        "maxiter": 300000,
        "return_eigenvectors": True,
    }
    if initial_guess is not None:
        eigsh_kwargs["v0"] = np.asarray(initial_guess, dtype=float)
    try:
        evals, evecs = eigsh(operator, **eigsh_kwargs)
        order = np.argsort(np.asarray(evals, dtype=float))
        idx = int(order[0])
        return {
            "electronic_energy_eh": float(np.asarray(evals, dtype=float)[idx]),
            "solver_kind": "sparse_eigsh",
            "subspace_dim": dim,
            "hamiltonian_nnz": None,
            "lowest_eigenvector": np.asarray(evecs[:, idx], dtype=float),
        }
    except Exception:
        H = np.zeros((dim, dim), dtype=float)
        for col in range(dim):
            basis_vec = np.zeros(dim, dtype=float)
            basis_vec[col] = 1.0
            H[:, col] = _restricted_matvec(basis_vec)
        H = 0.5 * (H + H.T)
        evals, evecs = np.linalg.eigh(H)
        return {
            "electronic_energy_eh": float(evals[0]),
            "solver_kind": "dense_eigh_fallback",
            "subspace_dim": dim,
            "hamiltonian_nnz": int(np.count_nonzero(np.abs(H) > 1.0e-14)),
            "lowest_eigenvector": np.asarray(evecs[:, 0], dtype=float),
        }


def _top_parent_detkeys_from_exactized(
    *,
    support_keys: list[DetKey],
    coeffs: np.ndarray,
    topn: int,
) -> list[DetKey]:
    if len(support_keys) != int(coeffs.shape[0]):
        raise ValueError("Exactized coefficient vector length does not match support size")
    ranked = sorted(
        zip(support_keys, np.asarray(coeffs, dtype=float), strict=False),
        key=lambda item: (abs(float(item[1])), int(item[0][0]), int(item[0][1])),
        reverse=True,
    )
    return [key for key, _coeff in ranked[: max(1, int(topn))]]


def _one_shell_support_from_parents(
    *,
    parents: list[DetKey],
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> set[DetKey]:
    from pyscf.fci import selected_ci

    shell: set[DetKey] = set()
    for parent in parents:
        parent_key = (int(parent[0]), int(parent[1]))
        parent_support = _enumerate_connected_sd_support({parent_key}, norb)
        if not parent_support:
            shell.add(parent_key)
            continue
        sci_parent = SparseCIVector({parent_key: 1.0}).to_scivector_on_support(parent_support)
        h_parent = selected_ci.contract_2e(h2eff, sci_parent, norb, nelec)
        shell.update((int(stra), int(strb)) for stra, strb in sparse_from_scivector(h_parent).coeffs)
        shell.add(parent_key)
    return shell


def _exactizeplus_support_union(
    *,
    support_keys: list[DetKey],
    exactized_coeffs: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    top_parents: int,
) -> dict[str, object]:
    if len(support_keys) == 0:
        raise ValueError("Cannot exactizeplus an empty determinant support")
    parent_keys = _top_parent_detkeys_from_exactized(
        support_keys=support_keys,
        coeffs=exactized_coeffs,
        topn=top_parents,
    )
    shell_support = _one_shell_support_from_parents(
        parents=parent_keys,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
    )
    base_support = {(int(stra), int(strb)) for stra, strb in support_keys}
    expanded_support = sorted(base_support | shell_support, key=lambda key: (int(key[0]), int(key[1])))
    exactizeplus = _exactize_support_union(
        support_keys=expanded_support,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        initial_guess=None,
    )
    exactizeplus["parent_count"] = int(len(parent_keys))
    exactizeplus["shell_added"] = int(len(set(expanded_support) - base_support))
    return exactizeplus


def _exactized_trace_payload(
    *,
    Q: list[SparseCIVector],
    params: CCIKThickRestartParams,
    cum_new_krylov: int,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    projected_eigvecs: np.ndarray | None,
) -> dict[str, object]:
    support_keys = _support_union_keys(Q)
    payload: dict[str, object] = {
        "support_union_size": int(len(support_keys)),
        "exactized_electronic_energy_eh": None,
        "exactized_solver_kind": None,
        "exactized_subspace_dim": None,
        "exactized_hamiltonian_nnz": None,
        "exactizeplus_electronic_energy_eh": None,
        "exactizeplus_solver_kind": None,
        "exactizeplus_subspace_dim": None,
        "exactizeplus_parent_count": None,
        "exactizeplus_shell_added": None,
    }
    period = _checkpoint_period_from_params(params)
    if period is None or int(cum_new_krylov) <= 0 or int(cum_new_krylov) % int(period) != 0:
        return payload

    guess = _restricted_ground_state_guess(Q, projected_eigvecs, support_keys)
    exactized = _exactize_support_union(
        support_keys=support_keys,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        initial_guess=guess,
    )
    payload["exactized_electronic_energy_eh"] = float(exactized["electronic_energy_eh"])
    payload["exactized_solver_kind"] = str(exactized["solver_kind"])
    payload["exactized_subspace_dim"] = int(exactized["subspace_dim"])
    payload["exactized_hamiltonian_nnz"] = (
        None if exactized["hamiltonian_nnz"] is None else int(exactized["hamiltonian_nnz"])
    )
    if _exactizeplus_enabled(params):
        exactizeplus = _exactizeplus_support_union(
            support_keys=support_keys,
            exactized_coeffs=np.asarray(exactized["lowest_eigenvector"], dtype=float),
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            top_parents=_exactizeplus_top_parents(params),
        )
        payload["exactizeplus_electronic_energy_eh"] = float(exactizeplus["electronic_energy_eh"])
        payload["exactizeplus_solver_kind"] = str(exactizeplus["solver_kind"])
        payload["exactizeplus_subspace_dim"] = int(exactizeplus["subspace_dim"])
        payload["exactizeplus_parent_count"] = int(exactizeplus["parent_count"])
        payload["exactizeplus_shell_added"] = int(exactizeplus["shell_added"])
    return payload



def gram_schmidt_list(vecs: list[SparseCIVector], tol: float = 1e-12) -> list[SparseCIVector]:
    """Orthonormalize sparse Krylov vectors and drop nearly dependent ones."""
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
    compression_policy = compression_policy_from_params(params, default="fixed_budget")
    if compression_policy == "threshold":
        unit = normalize_sparse(vec)
        out, _, _ = threshold_prune_sparse(unit, float(getattr(params, "store_abs_tol", 0.0)), keep_largest_if_empty=True)
        return out
    if compression_policy == "fixed_budget":
        return compress_topk_sparse_grouped(vec, nkeep=max(1, int(getattr(params, "fixed_store_keep", 1_000_000))))
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
    projected = solve_projected_ritz(
        Q=Q,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        projected_ritz_mode=projected_ritz_mode_from_params(params, default="exact_on_basis"),
    )
    return projected.evals, projected.eigvecs



def _ccik_cycle_sparse_multi_start(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    start_vecs: list[SparseCIVector],
    params: CCIKThickRestartParams,
    hdiag_full: dict[tuple[int, int], float] | None,
    cycle_index: int,
    cum_new_krylov_start: int = 0,
    iteration_callback: Callable[[ThickRestartIterationRecord], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[SparseCIVector], int]:
    """One thick-restart cycle: build a compressed sparse Krylov basis from multiple starts."""

    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]

    Q = gram_schmidt_list(start_vecs, tol=float(getattr(params, "orth_tol", 1e-12)))
    if len(Q) == 0:
        raise RuntimeError("All start vectors vanished after orthogonalization.")

    restart_seed_count = int(len(Q))
    cum_new_krylov = int(cum_new_krylov_start)
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
        cum_new_krylov += 1
        if iteration_callback is not None:
            evals_step, X_step = _cycle_ritz(
                Q=Q,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
            diagnostics = _exactized_trace_payload(
                Q=Q,
                params=params,
                cum_new_krylov=cum_new_krylov,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                projected_eigvecs=X_step,
            )
            _maybe_trace_iteration(
                iteration_callback,
                cycle=cycle_index,
                basis_size=len(Q),
                restart_seed_count=restart_seed_count,
                cum_new_krylov=cum_new_krylov,
                electronic_energy_eh=float(evals_step[0]),
                support_union_size=int(diagnostics["support_union_size"]),
                exactized_electronic_energy_eh=(
                    None
                    if diagnostics["exactized_electronic_energy_eh"] is None
                    else float(diagnostics["exactized_electronic_energy_eh"])
                ),
                exactized_solver_kind=(
                    None
                    if diagnostics["exactized_solver_kind"] is None
                    else str(diagnostics["exactized_solver_kind"])
                ),
                exactized_subspace_dim=(
                    None
                    if diagnostics["exactized_subspace_dim"] is None
                    else int(diagnostics["exactized_subspace_dim"])
                ),
                exactized_hamiltonian_nnz=(
                    None
                    if diagnostics["exactized_hamiltonian_nnz"] is None
                    else int(diagnostics["exactized_hamiltonian_nnz"])
                ),
                exactizeplus_electronic_energy_eh=(
                    None
                    if diagnostics["exactizeplus_electronic_energy_eh"] is None
                    else float(diagnostics["exactizeplus_electronic_energy_eh"])
                ),
                exactizeplus_solver_kind=(
                    None
                    if diagnostics["exactizeplus_solver_kind"] is None
                    else str(diagnostics["exactizeplus_solver_kind"])
                ),
                exactizeplus_subspace_dim=(
                    None
                    if diagnostics["exactizeplus_subspace_dim"] is None
                    else int(diagnostics["exactizeplus_subspace_dim"])
                ),
                exactizeplus_parent_count=(
                    None
                    if diagnostics["exactizeplus_parent_count"] is None
                    else int(diagnostics["exactizeplus_parent_count"])
                ),
                exactizeplus_shell_added=(
                    None
                    if diagnostics["exactizeplus_shell_added"] is None
                    else int(diagnostics["exactizeplus_shell_added"])
                ),
            )
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
    return evals, X, Q, cum_new_krylov



def ccik_ground_energy_dense_thick_restart(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKThickRestartParams | None = None,
    stats: dict[str, object] | None = None,
    iteration_callback: Callable[[ThickRestartIterationRecord], None] | None = None,
) -> float:
    """Run sparse CCIK with thick restart.

    Plain-language purpose:
    - provide the repository's most stable general-purpose solver

    Chemistry meaning:
    - keep several good Ritz vectors at restart instead of restarting from only
      one direction

    Inputs:
    - active-space integrals and electron counts
    - optional ``params`` dataclass

    Outputs:
    - the electronic CAS energy in Hartree
    """

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
            if iteration_callback is not None:
                evals_step, X_step = _cycle_ritz(
                    Q=Q,
                    h1=h1,
                    eri8=eri8,
                    h2eff=h2eff,
                    norb=norb,
                    nelec=nelec,
                    params=params,
                    hdiag_full=hdiag_full,
                )
                diagnostics = _exactized_trace_payload(
                    Q=Q,
                    params=params,
                    cum_new_krylov=len(Q) - 1,
                    h2eff=h2eff,
                    norb=norb,
                    nelec=nelec,
                    projected_eigvecs=X_step,
                )
                _maybe_trace_iteration(
                    iteration_callback,
                    cycle=1,
                    basis_size=len(Q),
                    restart_seed_count=1,
                    cum_new_krylov=len(Q) - 1,
                    electronic_energy_eh=float(evals_step[0]),
                    support_union_size=int(diagnostics["support_union_size"]),
                    exactized_electronic_energy_eh=(
                        None
                        if diagnostics["exactized_electronic_energy_eh"] is None
                        else float(diagnostics["exactized_electronic_energy_eh"])
                    ),
                    exactized_solver_kind=(
                        None
                        if diagnostics["exactized_solver_kind"] is None
                        else str(diagnostics["exactized_solver_kind"])
                    ),
                    exactized_subspace_dim=(
                        None
                        if diagnostics["exactized_subspace_dim"] is None
                        else int(diagnostics["exactized_subspace_dim"])
                    ),
                    exactized_hamiltonian_nnz=(
                        None
                        if diagnostics["exactized_hamiltonian_nnz"] is None
                        else int(diagnostics["exactized_hamiltonian_nnz"])
                    ),
                    exactizeplus_electronic_energy_eh=(
                        None
                        if diagnostics["exactizeplus_electronic_energy_eh"] is None
                        else float(diagnostics["exactizeplus_electronic_energy_eh"])
                    ),
                    exactizeplus_solver_kind=(
                        None
                        if diagnostics["exactizeplus_solver_kind"] is None
                        else str(diagnostics["exactizeplus_solver_kind"])
                    ),
                    exactizeplus_subspace_dim=(
                        None
                        if diagnostics["exactizeplus_subspace_dim"] is None
                        else int(diagnostics["exactizeplus_subspace_dim"])
                    ),
                    exactizeplus_parent_count=(
                        None
                        if diagnostics["exactizeplus_parent_count"] is None
                        else int(diagnostics["exactizeplus_parent_count"])
                    ),
                    exactizeplus_shell_added=(
                        None
                        if diagnostics["exactizeplus_shell_added"] is None
                        else int(diagnostics["exactizeplus_shell_added"])
                    ),
                )

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
    cum_new_krylov = 0

    for cyc in range(params.ncycles):
        evals, X, Q, cum_new_krylov = _ccik_cycle_sparse_multi_start(
            h1,
            eri8,
            norb,
            nelec,
            start_vecs=starts,
            params=params,
            hdiag_full=hdiag_full,
            cycle_index=cyc + 1,
            cum_new_krylov_start=cum_new_krylov,
            iteration_callback=iteration_callback,
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
