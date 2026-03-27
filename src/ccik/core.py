from __future__ import annotations

import numpy as np

from .params import CCIKParams
from .propagation import build_next_krylov_vector, format_krylov_step_diagnostics, krylov_step_diagnostics_as_dict
from .selection import build_projection_workspace, candidate_mode_from_params, make_full_hdiag_map
from .sparse_ci import (
    SparseCIVector,
    inner_sparse,
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
    nrm = np.linalg.norm(ci.ravel())
    return ci if nrm == 0 else (ci / nrm)



def inner(ci1: np.ndarray, ci2: np.ndarray) -> float:
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
    s = 0
    for p in occ:
        s |= (1 << p)
    return s



def generalized_eigh(H: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Solve H x = e S x for symmetric H and positive-semidefinite S.

    Compression can make the Krylov basis nearly linearly dependent, especially
    in threshold mode with weak truncation. Project small-overlap directions out
    of the generalized eigenproblem instead of relying on a fragile Cholesky of S.
    """
    H = 0.5 * (H + H.T)
    S = 0.5 * (S + S.T)

    s_eval, U = np.linalg.eigh(S)
    s_scale = float(max(np.max(np.abs(s_eval)), 1.0))
    keep = s_eval > float(eps) * s_scale
    if not np.any(keep):
        keep = s_eval > float(eps)
    if not np.any(keep):
        raise np.linalg.LinAlgError("Generalized Ritz overlap matrix is numerically singular.")

    s_keep = s_eval[keep]
    U_keep = U[:, keep]
    Sinv_half = U_keep / np.sqrt(s_keep)
    A = Sinv_half.T @ H @ Sinv_half
    A = 0.5 * (A + A.T)
    evals, Y = np.linalg.eigh(A)
    X = Sinv_half @ Y
    return evals, X



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
    evals, _ = generalized_eigh(Hproj, Sproj)
    return float(evals[0]), Hproj, Sproj



def ccik_ground_energy_dense(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKParams | None = None,
    stats: dict[str, object] | None = None,
) -> float:
    """Selected-space CCIK using sparse Krylov vectors over determinant pairs."""

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
