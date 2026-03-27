from __future__ import annotations

import numpy as np

from .core import generalized_eigh
from .params import CCIKStochasticParams
from .propagation import build_next_krylov_vector, format_krylov_step_diagnostics, krylov_step_diagnostics_as_dict
from .selection import build_projection_workspace, candidate_mode_from_params, make_full_hdiag_map
from .sparse_ci import (
    SparseCIVector,
    inner_sparse,
    make_hf_sparse_vector,
)



def ccik_ground_energy_stochastic(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKStochasticParams | None = None,
    stats: dict[str, object] | None = None,
) -> float:
    """Run CCIK with stochastic multidepth candidate discovery over sparse determinant supports."""

    if params is None:
        params = CCIKStochasticParams()

    from pyscf.fci import direct_spin1

    mode = candidate_mode_from_params(params, default="stochastic")
    hdiag_full: dict[tuple[int, int], float] | None = None
    if mode == "full":
        hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)
    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=norb, nelec=nelec)

    rng = np.random.default_rng(None if params.seed is None else int(params.seed))
    Q: list[SparseCIVector] = [q0]
    if stats is not None:
        stats["step_diagnostics"] = []

    for k in range(int(params.m) - 1):
        qk = Q[k]
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
            rng=rng,
        )
        if params.verbose:
            print(format_krylov_step_diagnostics(f"    k={k:02d}", diag))
        if stats is not None:
            step_diags = stats.setdefault("step_diagnostics", [])
            if isinstance(step_diags, list):
                step_diags.append(krylov_step_diagnostics_as_dict(diag))
            stats["last_step_diagnostics"] = krylov_step_diagnostics_as_dict(diag)
        if not q_next.coeffs:
            if params.verbose:
                print(f"[CCIK-STOCHASTIC] breakdown at k={k}")
            break
        Q.append(q_next)

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
            rng=rng,
        ).v_proj
        for j in range(m_eff)
    ]
    for i in range(m_eff):
        for j in range(m_eff):
            Sproj[i, j] = inner_sparse(Q[i], Q[j])
            Hproj[i, j] = inner_sparse(Q[i], Hq[j])

    evals, _ = generalized_eigh(Hproj, Sproj)

    if stats is not None:
        union = set()
        ndet_sum = 0
        for q in Q:
            supp = q.support()
            union.update(supp)
            ndet_sum += len(supp)
        stats["m_eff"] = int(m_eff)
        stats["ndet_sum"] = int(ndet_sum)
        stats["ndet_union"] = int(len(union))

    return float(evals[0])
