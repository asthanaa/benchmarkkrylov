"""Shared projected-Ritz helpers used by the deterministic sparse solvers.

This module centralizes the projected Hamiltonian construction so both the
solver loops and selector logic can work with the same Krylov-space data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .selection import build_candidate_pool_impl, build_projection_workspace, compute_projected_matvec_impl
from .sparse_ci import SparseCIVector, inner_sparse, sparse_from_dense

DetKey = tuple[int, int]


def projected_ritz_mode_from_params(params: Any, *, default: str = "approximate") -> str:
    raw = str(getattr(params, "projected_ritz_mode", default)).strip().lower().replace("-", "_")
    aliases = {
        "exact": "exact_on_basis",
        "exact_on_basis": "exact_on_basis",
        "exact_basis": "exact_on_basis",
        "approximate": "approximate",
        "approx": "approximate",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            f"Unknown projected_ritz_mode {raw!r}. Expected one of: exact_on_basis, approximate."
        ) from exc


@dataclass(frozen=True)
class ProjectedRitzData:
    """Projected Ritz solve data for a fixed sparse Krylov basis."""

    evals: np.ndarray
    eigvecs: np.ndarray
    Hproj: np.ndarray
    Sproj: np.ndarray
    projected_columns: tuple[SparseCIVector, ...]


def generalized_eigh(H: np.ndarray, S: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``H x = e S x`` for symmetric ``H`` and positive-semidefinite ``S``."""

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


def build_projected_columns(
    Q: list[SparseCIVector],
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    candidate_pool_override: set[DetKey] | None = None,
) -> tuple[SparseCIVector, ...]:
    """Build sparse projected ``H q_j`` columns for every basis vector in ``Q``."""

    columns: list[SparseCIVector] = []
    for qj in Q:
        if candidate_pool_override is None:
            ws = build_projection_workspace(
                qk=qj,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
        else:
            parent_support, _ = build_candidate_pool_impl(
                qk=qj,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
            ws = compute_projected_matvec_impl(
                qk=qj,
                candidate_pool=set(candidate_pool_override),
                parent_support=parent_support,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
        columns.append(ws.v_proj)
    return tuple(columns)


def build_projected_matrices_from_columns(
    Q: list[SparseCIVector],
    projected_columns: tuple[SparseCIVector, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble projected ``H`` and overlap matrices from a fixed basis and columns."""

    m_eff = len(Q)
    Hproj = np.zeros((m_eff, m_eff))
    Sproj = np.zeros((m_eff, m_eff))
    for i in range(m_eff):
        for j in range(m_eff):
            Sproj[i, j] = inner_sparse(Q[i], Q[j])
            Hproj[i, j] = inner_sparse(Q[i], projected_columns[j])
    return Hproj, Sproj


def solve_projected_ritz(
    *,
    Q: list[SparseCIVector],
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    candidate_pool_override: set[DetKey] | None = None,
    projected_ritz_mode: str = "approximate",
) -> ProjectedRitzData:
    """Build the projected Krylov matrices and solve the generalized Ritz problem."""

    mode = projected_ritz_mode_from_params(params, default=projected_ritz_mode)
    if mode == "exact_on_basis":
        Hproj, Sproj, projected_columns = build_exact_projected_matrices(
            Q,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
        )
    else:
        projected_columns = build_projected_columns(
            Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
            candidate_pool_override=candidate_pool_override,
        )
        Hproj, Sproj = build_projected_matrices_from_columns(Q, projected_columns)
    evals, eigvecs = generalized_eigh(Hproj, Sproj)
    return ProjectedRitzData(
        evals=evals,
        eigvecs=eigvecs,
        Hproj=Hproj,
        Sproj=Sproj,
        projected_columns=projected_columns,
    )


def build_exact_projected_matrices(
    Q: list[SparseCIVector],
    *,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[SparseCIVector, ...]]:
    """Reevaluate the projected Hamiltonian exactly on the current sparse basis."""

    from pyscf.fci import direct_spin1

    exact_columns: list[SparseCIVector] = []
    for qj in Q:
        dense_q = qj.to_dense(norb=norb, nelec=nelec)
        dense_v = np.asarray(direct_spin1.contract_2e(h2eff, dense_q, norb, nelec), dtype=float)
        exact_columns.append(sparse_from_dense(dense_v, norb=norb, nelec=nelec))
    columns = tuple(exact_columns)
    Hproj, Sproj = build_projected_matrices_from_columns(Q, columns)
    return Hproj, Sproj, columns
