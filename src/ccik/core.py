"""Baseline deterministic CCIK solver and core linear-algebra helpers.

This file is the best place to understand the basic compressed-Krylov loop
before studying thick restart or stochastic discovery.
"""

from __future__ import annotations

import numpy as np

from .params import CCIKParams
from .propagation import build_next_krylov_vector, format_krylov_step_diagnostics, krylov_step_diagnostics_as_dict
from .ritz import generalized_eigh, projected_ritz_mode_from_params, solve_projected_ritz
from .selection import candidate_mode_from_params, make_full_hdiag_map
from .sparse_ci import (
    SparseCIVector,
    make_hf_sparse_vector,
)


# PDF mapping note
# ---------------
# This module implements the “Compressed Krylov Subspace Method with Selected-Configuration Matvecs”
# described in `krylov_cipsi.pdf`, Section II (A–F).
#
# Key correspondences used below:
# - Krylov space: PDF Sec. II.A, Eq. (1)
# - Exact matvec coefficients v_I^(k): PDF Sec. II.B, Eq. (4)
# - Rayleigh quotient E_k: PDF Sec. II.C, Eq. (5)
# - Selection score eta_I^(k): PDF Sec. II.C, Eq. (6)
# - Compressed matvec |v_tilde_k> = P_{S_k} H|q_k>: PDF Sec. II.C, Eq. (7)–(8)
# - Orthogonalization / next basis vector: PDF Sec. II.E, Eq. (11)–(12)
# - Ritz in Krylov subspace: PDF Sec. II.F, Eq. (14)–(17)
#
# Important implementation detail:
# The PDF assumes exact orthonormality so S=I (PDF Eq. (13)). Here we also compress vectors, so we
# solve a generalized Ritz problem using the computed overlap Sproj.


def normalize(ci: np.ndarray) -> np.ndarray:
    """Return a normalized dense CI array, leaving an all-zero array unchanged."""

    nrm = np.linalg.norm(ci.ravel())
    return ci if nrm == 0 else (ci / nrm)



def inner(ci1: np.ndarray, ci2: np.ndarray) -> float:
    """Return the usual dense-vector inner product."""

    return float(np.vdot(ci1.ravel(), ci2.ravel()))



def compress_keep_top_mask(ci: np.ndarray, nkeep: int | None = None, min_abs: float = 0.0) -> np.ndarray:
    """Boolean mask keeping up to `nkeep` largest |coeff| entries (and >= `min_abs`)."""
    flat = ci.ravel()
    absflat = np.abs(flat)

    if nkeep is None or nkeep >= flat.size:
        return (absflat >= min_abs).reshape(ci.shape)

    eligible = np.where(absflat > min_abs)[0]
    if eligible.size == 0:
        idx = int(np.argmax(absflat))
        mask = np.zeros_like(flat, dtype=bool)
        mask[idx] = True
        return mask.reshape(ci.shape)

    if eligible.size > nkeep:
        sub = absflat[eligible]
        top_idx = np.argpartition(sub, -nkeep)[-nkeep:]
        keep = eligible[top_idx]
    else:
        keep = eligible

    mask = np.zeros_like(flat, dtype=bool)
    mask[keep] = True
    return mask.reshape(ci.shape)



def apply_mask(ci: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(ci)
    out[mask] = ci[mask]
    return out



def topk_positive_flat_indices(arr: np.ndarray, k: int) -> np.ndarray:
    """Return flat indices of up to k entries with arr>0 and largest values.

    The returned indices are not sorted.
    """

    if k <= 0:
        return np.asarray([], dtype=int)

    flat = np.asarray(arr).ravel()
    pos = np.where(flat > 0)[0]
    if pos.size == 0:
        return np.asarray([], dtype=int)

    nsel = min(int(k), int(pos.size))
    vals = flat[pos]
    top_local = np.argpartition(vals, -nsel)[-nsel:]
    return pos[top_local]



def topk_positive_mask(arr: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean mask selecting top-k entries with arr>0."""

    flat_idx = topk_positive_flat_indices(arr, int(k))
    mask = np.zeros_like(np.asarray(arr), dtype=bool)
    if flat_idx.size == 0:
        return mask
    mask.ravel()[flat_idx] = True
    return mask



def occ_list_to_bitstring(occ: list[int]) -> int:
    """Convert an occupied-orbital list into the bitstring format used by PySCF."""

    s = 0
    for p in occ:
        s |= (1 << p)
    return s



def _ritz_energy(
    *,
    Q: list[SparseCIVector],
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: CCIKParams,
    hdiag_full: dict[tuple[int, int], float] | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Solve the projected Ritz problem for the current sparse Krylov basis."""

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
    return float(projected.evals[0]), projected.Hproj, projected.Sproj



def ccik_ground_energy_dense(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKParams | None = None,
    stats: dict[str, object] | None = None,
) -> float:
    """Run the baseline sparse CCIK solver.

    Plain-language purpose:
    - estimate the ground-state CAS electronic energy with the deterministic
      baseline solver

    Chemistry meaning:
    - builds a compressed Krylov basis over determinant pairs and performs a
      Ritz solve in that subspace

    Inputs:
    - active-space integrals and electron counts
    - optional ``params`` dataclass

    Outputs:
    - the electronic CAS energy in Hartree
    """

    if params is None:
        params = CCIKParams()

    from pyscf.fci import direct_spin1

    mode = candidate_mode_from_params(params, default="selected_ci")
    hdiag_full: dict[tuple[int, int], float] | None = None
    if mode == "full":
        hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=norb, nelec=nelec)

    Q: list[SparseCIVector] = [q0]

    if stats is not None:
        stats["step_diagnostics"] = []

    for k in range(params.m - 1):
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
                print(f"[CCIK] breakdown at k={k}")
            break
        Q.append(q_next)

    E0, _, _ = _ritz_energy(
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
        stats["m_eff"] = int(len(Q))
        stats["ndet_union"] = int(len(union))
        stats["ndet_sum"] = int(ndet_sum)

    return float(E0)
