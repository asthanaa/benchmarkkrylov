"""Turn a projected sparse ``H q_k`` into the next compressed Krylov vector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .selection import (
    build_candidate_pool_impl,
    candidate_mode_from_params,
    compression_policy_from_params,
    compute_projected_matvec_impl,
    selector_mode_from_params,
)
from .ritz import build_projected_columns, solve_projected_ritz
from .sparse_ci import (
    SparseCIVector,
    SparseProjectionWorkspace,
    add_scaled_sparse,
    compress_topk_sparse,
    compress_topk_sparse_grouped,
    compress_with_support_sparse,
    inner_sparse,
    linear_combination_sparse,
    normalize_sparse,
    orthogonalize_sparse,
    project_sparse_to_support,
    sparse_norm,
    sparse_norm_sq,
    threshold_prune_sparse,
    topk_abs_keys_sparse,
)

DetKey = tuple[int, int]
_THRESHOLD_MAX_STORED = 2_000_000


def _fixed_store_keep(params: Any) -> int:
    return max(1, int(getattr(params, "fixed_store_keep", 1_000_000)))


@dataclass(frozen=True)
class KrylovStepDiagnostics:
    """Numerical summary of one sparse Krylov update."""

    q_norm: float
    hq_norm: float
    projected_matvec_norm: float
    discarded_weight: float
    orthogonalized_residual_norm: float
    post_compression_residual_norm: float
    max_basis_overlap: float
    selector_mode_used: str
    candidate_mode_used: str
    num_candidates_generated: int
    num_tmp_kept: int
    num_final_kept: int
    matvec_error_norm: float | None
    relative_matvec_error: float | None
    compression_policy_used: str
    propagation_support_size: int
    propagation_dropped_norm: float
    propagation_used_fallback: bool
    store_dropped_norm: float
    store_used_fallback: bool
    selector_reference_energy: float | None = None
    probe_residual_norm: float | None = None
    rebuilt_residual_norm: float | None = None
    propagation_score_mode: str | None = None
    storage_score_mode: str | None = None
    target_ritz_energy: float | None = None
    target_basis_size: int | None = None
    selector_score_mode: str | None = None
    next_propagation_support_size: int | None = None
    delayed_propagation_used_fallback: bool | None = None


@dataclass(frozen=True)
class KrylovStepArtifacts:
    """Intermediate sparse objects produced during one Krylov step."""

    requested_selector: str
    used_selector: str
    score_vector: SparseCIVector
    v_temp: SparseCIVector
    orthogonalized_residual: SparseCIVector
    compressed: SparseCIVector
    tmp_keep: int
    preserve_cap: int
    preserve_support: set[DetKey]
    candidate_pool: set[DetKey]
    parent_support: set[DetKey]
    propagation_score_support: set[DetKey]
    score_support: set[DetKey]


def build_candidate_pool(
    q_k: SparseCIVector,
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    rng: np.random.Generator | None = None,
) -> set[DetKey]:
    _, candidate_pool = build_candidate_pool_impl(
        qk=q_k,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        rng=rng,
    )
    return candidate_pool


def compute_projected_matvec(
    q_k: SparseCIVector,
    candidate_indices: set[DetKey],
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    rng: np.random.Generator | None = None,
) -> SparseProjectionWorkspace:
    parent_support, actual_pool = build_candidate_pool_impl(
        qk=q_k,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        rng=rng,
    )
    if candidate_indices:
        actual_pool = set(candidate_indices)
    return compute_projected_matvec_impl(
        qk=q_k,
        candidate_pool=actual_pool,
        parent_support=parent_support,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )


def orthogonalize_against_basis(
    v_proj: SparseCIVector,
    basis: list[SparseCIVector],
    S_metric=None,
) -> SparseCIVector:
    _ = S_metric
    return orthogonalize_sparse(v_proj, basis)


def compress_vector(
    vec: SparseCIVector,
    nkeep: int,
    preserve_support: set[DetKey] | None = None,
) -> SparseCIVector:
    if preserve_support:
        return compress_with_support_sparse(vec, nkeep, preserve_support=preserve_support)
    return compress_topk_sparse(vec, nkeep=nkeep)


def krylov_step_diagnostics_as_dict(diag: KrylovStepDiagnostics) -> dict[str, object]:
    return asdict(diag)


def format_krylov_step_diagnostics(step_label: str, diag: KrylovStepDiagnostics) -> str:
    msg = (
        f"{step_label}: policy={diag.compression_policy_used} selector={diag.selector_mode_used} candidate={diag.candidate_mode_used} "
        f"|q|={diag.q_norm:.6e} |Hq|≈{diag.hq_norm:.6e} |v_proj|={diag.projected_matvec_norm:.6e} "
        f"drop={diag.discarded_weight:.6e} |r_orth|={diag.orthogonalized_residual_norm:.6e} "
        f"|r_keep|={diag.post_compression_residual_norm:.6e} overlap={diag.max_basis_overlap:.3e} "
        f"cand={diag.num_candidates_generated} tmp={diag.num_tmp_kept} final={diag.num_final_kept}"
    )
    if diag.compression_policy_used == "threshold":
        msg += (
            f" prop_supp={diag.propagation_support_size} prop_drop={diag.propagation_dropped_norm:.3e}"
            f" store_drop={diag.store_dropped_norm:.3e}"
        )
        if diag.propagation_used_fallback:
            msg += " prop_fallback=1"
        if diag.store_used_fallback:
            msg += " store_fallback=1"
    elif diag.compression_policy_used == "fixed_budget":
        msg += f" prop_supp={diag.propagation_support_size}"
    if diag.relative_matvec_error is not None:
        msg += f" rel_matvec_err={diag.relative_matvec_error:.3e}"
    return msg


def _runtime_preserve_prev_support_cap(params: Any) -> int:
    raw = getattr(params, "preserve_prev_support_cap", None)
    if raw is not None:
        return max(0, int(raw))
    return max(0, min(2048, max(1, int(getattr(params, "nkeep", 1))) // 4))


def _compression_policy(params: Any) -> str:
    return compression_policy_from_params(params, default="fixed_budget")


def _tmp_keep(params: Any) -> int:
    nkeep = max(1, int(getattr(params, "nkeep", 1)))
    nadd = max(0, int(getattr(params, "nadd", 0)))
    kv = max(0, int(getattr(params, "Kv", 0)))
    factor = max(1, int(getattr(params, "tmp_keep_factor", 4)))
    return max(factor * nkeep, nkeep + nadd + kv)


def _topk_score_keys(scores: dict[DetKey, float], k: int) -> set[DetKey]:
    if k <= 0:
        return set()
    ranked = sorted(((key, float(val)) for key, val in scores.items() if float(val) > 0.0), key=lambda item: item[1], reverse=True)
    return {key for key, _ in ranked[: int(k)]}


def _legacy_score_map(
    *,
    qk: SparseCIVector,
    v_proj: SparseCIVector,
    hdiag: dict[DetKey, float],
    candidate_support: set[DetKey],
    eps_denom: float,
) -> tuple[float, dict[DetKey, float]]:
    Ek = inner_sparse(qk, v_proj)
    score: dict[DetKey, float] = {}
    for det in candidate_support:
        amp = v_proj.get(det, 0.0)
        if amp == 0.0:
            continue
        denom = max(abs(Ek - hdiag.get(det, Ek)), eps_denom)
        score[det] = (abs(amp) ** 2) / denom
    return Ek, score


def _pt2_score_map(
    *,
    vec: SparseCIVector,
    hdiag: dict[DetKey, float],
    energy: float,
    eps_denom: float,
) -> dict[DetKey, float]:
    score: dict[DetKey, float] = {}
    for det, amp in vec.coeffs.items():
        if amp == 0.0:
            continue
        denom = max(abs(energy - hdiag.get(det, energy)), eps_denom)
        score[det] = (abs(amp) ** 2) / denom
    return score


def _score_from_vector(vec: SparseCIVector, mode: str) -> dict[DetKey, float]:
    if mode in {"abs_hq", "abs_residual"}:
        return {key: abs(val) for key, val in vec.coeffs.items() if abs(val) > 0.0}
    if mode in {"sq_hq", "sq_residual"}:
        return {key: abs(val) ** 2 for key, val in vec.coeffs.items() if abs(val) > 0.0}
    raise ValueError(f"Unsupported score mode {mode!r}")


def _fixed_budget_keep(params: Any) -> int:
    return max(1, int(getattr(params, "fixed_store_keep", 1_000_000)))


def _fixed_budget_propagate_keep(params: Any) -> int:
    return max(1, int(getattr(params, "fixed_propagate_keep", 10_000)))


def _fixed_budget_score_support(
    *,
    qk: SparseCIVector,
    basis: list[SparseCIVector],
    ws: SparseProjectionWorkspace,
    orth_residual: SparseCIVector,
    used_selector: str,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None,
) -> tuple[set[DetKey], SparseCIVector, float | None, int | None, str]:
    keep = _fixed_budget_keep(params)
    target_ritz_energy: float | None = None
    target_basis_size: int | None = None
    selector_score_mode = used_selector

    if used_selector == "pt2_residual":
        raise ValueError("pt2_residual uses a dedicated fixed-budget single-pass path.")
    if used_selector == "pt2_legacy":
        _, legacy_scores = _legacy_score_map(
            qk=qk,
            v_proj=ws.v_proj,
            hdiag=ws.hdiag,
            candidate_support=ws.candidate_pool,
            eps_denom=float(getattr(params, "eps_denom", 1e-12)),
        )
        support = _topk_score_keys(legacy_scores, keep)
        score_vec = ws.v_proj
    elif used_selector in {"abs_hq", "sq_hq"}:
        score_vec = ws.v_proj
        support = _topk_score_keys(_score_from_vector(score_vec, used_selector), keep)
    elif used_selector in {"abs_residual", "sq_residual"}:
        score_vec = orth_residual
        support = _topk_score_keys(_score_from_vector(score_vec, used_selector), keep)
    elif used_selector == "ground_state_pt2":
        ritz_basis = basis if basis else [qk]
        projected = solve_projected_ritz(
            Q=ritz_basis,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        coeffs = projected.eigvecs[:, 0]
        pool_columns = build_projected_columns(
            ritz_basis,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
            candidate_pool_override=ws.candidate_pool,
        )
        score_vec = project_sparse_to_support(
            linear_combination_sparse((float(coeffs[i]), pool_columns[i]) for i in range(len(ritz_basis))),
            ws.candidate_pool,
        )
        target_ritz_energy = float(projected.evals[0])
        target_basis_size = int(len(ritz_basis))
        score_map: dict[DetKey, float] = {}
        eps_denom = float(getattr(params, "eps_denom", 1e-12))
        for det in ws.candidate_pool:
            amp = score_vec.get(det, 0.0)
            if amp == 0.0:
                continue
            denom = max(abs(target_ritz_energy - ws.hdiag.get(det, target_ritz_energy)), eps_denom)
            score_map[det] = (abs(amp) ** 2) / denom
        support = _topk_score_keys(score_map, keep)
        selector_score_mode = "ground_state_pt2"
    else:
        raise ValueError(f"Unsupported fixed-budget selector mode {used_selector!r}")

    if not support:
        support = topk_abs_keys_sparse(orth_residual, keep)
    return support, score_vec, target_ritz_energy, target_basis_size, selector_score_mode


def _max_basis_overlap(vec: SparseCIVector, basis: list[SparseCIVector]) -> float:
    if not vec.coeffs or not basis:
        return 0.0
    return max(abs(inner_sparse(vec, q)) for q in basis)


def _secondortho_enabled(params: Any) -> bool:
    return bool(getattr(params, "secondortho", False))


def _support_restricted_second_orthogonalize(
    vec: SparseCIVector,
    basis: list[SparseCIVector],
    params: Any,
) -> SparseCIVector:
    """Reorthogonalize after Kstore compression without changing the support."""

    support = set(vec.coeffs)
    if not support or not basis:
        return vec.copy()

    orth_tol = max(0.0, float(getattr(params, "orth_tol", 1e-12)))
    norm_floor_sq = max(1.0e-30, orth_tol * orth_tol)
    projected_basis = [
        projected
        for q in basis
        if (projected := project_sparse_to_support(q, support)).coeffs and sparse_norm_sq(projected) > norm_floor_sq
    ]
    if not projected_basis:
        return vec.copy()

    m = len(projected_basis)
    gram = np.empty((m, m), dtype=float)
    rhs = np.empty(m, dtype=float)
    for i, qi in enumerate(projected_basis):
        rhs[i] = inner_sparse(qi, vec)
        for j in range(i, m):
            val = inner_sparse(qi, projected_basis[j])
            gram[i, j] = val
            gram[j, i] = val

    evals, evecs = np.linalg.eigh(0.5 * (gram + gram.T))
    max_eval = float(np.max(evals)) if evals.size else 0.0
    cutoff = max(1.0e-14, orth_tol * max(1.0, abs(max_eval)))
    keep = evals > cutoff
    if not bool(np.any(keep)):
        return vec.copy()

    rhs_eig = evecs[:, keep].T @ rhs
    coeffs = evecs[:, keep] @ (rhs_eig / evals[keep])
    out = vec.copy()
    for coeff, projected in zip(coeffs, projected_basis, strict=False):
        if abs(float(coeff)) > 0.0:
            out = add_scaled_sparse(out, -float(coeff), projected)
    return project_sparse_to_support(out, support)


def _analyze_fixed_budget_pt2_residual(
    *,
    qk: SparseCIVector,
    basis: list[SparseCIVector],
    ws: SparseProjectionWorkspace,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None,
    rng: np.random.Generator | None,
) -> tuple[SparseCIVector, SparseProjectionWorkspace, SparseCIVector, set[DetKey], set[DetKey], float]:
    _ = (h1, eri8, h2eff, norb, nelec, hdiag_full, rng)
    eps_denom = float(getattr(params, "eps_denom", 1e-12))
    orth_residual = orthogonalize_against_basis(ws.v_proj, basis)
    selector_energy = inner_sparse(qk, ws.v_proj)
    store_scores = _pt2_score_map(
        vec=orth_residual,
        hdiag=ws.hdiag,
        energy=selector_energy,
        eps_denom=eps_denom,
    )
    store_support = _topk_score_keys(store_scores, _fixed_budget_keep(params))
    if not store_support:
        store_support = topk_abs_keys_sparse(orth_residual, _fixed_budget_keep(params))
    compressed = project_sparse_to_support(orth_residual, store_support)
    next_propagation = normalize_sparse(compressed)
    propagate_scores = _pt2_score_map(
        vec=next_propagation,
        hdiag=ws.hdiag,
        energy=selector_energy,
        eps_denom=eps_denom,
    )
    propagation_support = _topk_score_keys(propagate_scores, _fixed_budget_propagate_keep(params))
    if not propagation_support:
        propagation_support = topk_abs_keys_sparse(next_propagation, _fixed_budget_propagate_keep(params))
    return (
        compressed,
        ws,
        orth_residual,
        propagation_support,
        store_support,
        selector_energy,
    )


def analyze_krylov_step(
    q_k: SparseCIVector,
    basis: list[SparseCIVector],
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SparseCIVector, KrylovStepDiagnostics, SparseProjectionWorkspace, KrylovStepArtifacts]:
    parent_support, candidate_pool = build_candidate_pool_impl(
        qk=q_k,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        rng=rng,
    )
    ws = compute_projected_matvec_impl(
        qk=q_k,
        candidate_pool=candidate_pool,
        parent_support=parent_support,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
    )

    compression_policy = _compression_policy(params)
    if compression_policy == "fixed_budget":
        # QP keeps the full projected Hq through orthogonalization and only applies
        # the fixed store budget when storing q_{k+1}.
        ws = SparseProjectionWorkspace(
            v_proj=ws.v_proj,
            hdiag=ws.hdiag,
            space=ws.space,
            candidate_support=ws.candidate_support,
            candidate_pool=ws.candidate_pool,
            parent_support=ws.parent_support,
            propagation_support=ws.propagation_support,
            propagation_dropped_norm=ws.propagation_dropped_norm,
            propagation_used_fallback=ws.propagation_used_fallback,
            exact_v=ws.exact_v,
        )

    requested_selector = selector_mode_from_params(params, default="abs_residual")
    used_selector = requested_selector
    score_vec = ws.v_proj
    target_ritz_energy: float | None = None
    target_basis_size: int | None = None
    selector_score_mode: str | None = requested_selector

    if requested_selector == "ground_state_pt2" and compression_policy != "fixed_budget":
        raise ValueError(
            "selector_mode='ground_state_pt2' is only supported when compression_policy='fixed_budget'."
        )
    if requested_selector == "ground_state_pt2" and candidate_mode_from_params(params, default="selected_ci") == "stochastic":
        raise ValueError("selector_mode='ground_state_pt2' is only supported for deterministic candidate discovery.")
    if requested_selector == "pt2_residual" and compression_policy != "fixed_budget":
        raise ValueError(
            "selector_mode='pt2_residual' is only supported when compression_policy='fixed_budget'."
        )
    if requested_selector == "pt2_residual" and candidate_mode_from_params(params, default="selected_ci") == "stochastic":
        raise ValueError("selector_mode='pt2_residual' is only supported for deterministic candidate discovery.")

    if requested_selector in {"abs_residual", "sq_residual"}:
        if basis:
            score_vec = orthogonalize_against_basis(ws.v_proj, basis)
        else:
            used_selector = "abs_hq" if requested_selector == "abs_residual" else "sq_hq"
            score_vec = ws.v_proj
        selector_score_mode = used_selector

    score_support: set[DetKey] = set()
    propagation_score_support: set[DetKey] = set()
    preserve_support: set[DetKey] = set()
    preserve_cap = 0
    store_dropped_norm = 0.0
    store_used_fallback = False
    orth_residual = orthogonalize_against_basis(ws.v_proj, basis)
    selector_reference_energy: float | None = None
    probe_residual_norm: float | None = None
    rebuilt_residual_norm: float | None = None
    propagation_score_mode: str | None = None
    storage_score_mode: str | None = None
    next_propagation_support_size: int | None = None
    delayed_propagation_used_fallback: bool | None = None

    if compression_policy == "threshold":
        v_temp = ws.v_proj.copy()
        discarded_weight = 0.0
        orth_unit = normalize_sparse(orth_residual)
        compressed, store_dropped_norm, store_used_fallback = threshold_prune_sparse(
            orth_unit,
            float(getattr(params, "store_abs_tol", 0.0)),
            keep_largest_if_empty=True,
        )
        if len(compressed.coeffs) > _THRESHOLD_MAX_STORED:
            raise RuntimeError(
                "threshold compression policy exceeded internal stored-support limit "
                f"({_THRESHOLD_MAX_STORED}); loosen thresholds or switch back to topk compression."
        )
        q_next = normalize_sparse(compressed)
        score_support = set(score_vec.coeffs)
    elif compression_policy == "fixed_budget":
        if used_selector == "pt2_residual":
            (
                compressed,
                ws,
                orth_residual,
                propagation_score_support,
                score_support,
                selector_reference_energy,
            ) = _analyze_fixed_budget_pt2_residual(
                qk=q_k,
                basis=basis,
                ws=ws,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
                rng=rng,
            )
            score_vec = orth_residual
            selector_score_mode = "pt2_residual"
            propagation_score_mode = "pt2_residual"
            storage_score_mode = "pt2_residual"
            probe_residual_norm = sparse_norm(orth_residual)
            rebuilt_residual_norm = sparse_norm(orth_residual)
            discarded_weight = float(np.sqrt(max(sparse_norm_sq(orth_residual) - sparse_norm_sq(compressed), 0.0)))
            v_temp = compressed.copy()
            store_dropped_norm = discarded_weight
            if _secondortho_enabled(params):
                compressed = _support_restricted_second_orthogonalize(compressed, basis, params)
                q_probe = normalize_sparse(compressed)
                propagate_scores = _pt2_score_map(
                    vec=q_probe,
                    hdiag=ws.hdiag,
                    energy=float(selector_reference_energy),
                    eps_denom=float(getattr(params, "eps_denom", 1e-12)),
                )
                propagation_score_support = _topk_score_keys(propagate_scores, _fixed_budget_propagate_keep(params))
                if not propagation_score_support:
                    propagation_score_support = topk_abs_keys_sparse(q_probe, _fixed_budget_propagate_keep(params))
            q_next = normalize_sparse(compressed)
        else:
            (
                score_support,
                score_vec,
                target_ritz_energy,
                target_basis_size,
                selector_score_mode,
            ) = _fixed_budget_score_support(
                qk=q_k,
                basis=basis,
                ws=ws,
                orth_residual=orth_residual,
                used_selector=used_selector,
                h1=h1,
                eri8=eri8,
                h2eff=h2eff,
                norb=norb,
                nelec=nelec,
                params=params,
                hdiag_full=hdiag_full,
            )
            # QP fixed-budget mode always stores the top-|orth_residual| entries, even
            # when selector-specific score supports are used for diagnostics.
            compressed = compress_topk_sparse_grouped(orth_residual, _fixed_budget_keep(params))
            discarded_weight = 0.0
            v_temp = ws.v_proj.copy()
            store_dropped_norm = float(np.sqrt(max(sparse_norm_sq(orth_residual) - sparse_norm_sq(compressed), 0.0)))
            storage_score_mode = "orth_residual_topk"
            if _secondortho_enabled(params):
                compressed = _support_restricted_second_orthogonalize(compressed, basis, params)
            q_next = normalize_sparse(compressed)
    else:
        if used_selector == "pt2_legacy":
            _, legacy_scores = _legacy_score_map(
                qk=q_k,
                v_proj=ws.v_proj,
                hdiag=ws.hdiag,
                candidate_support=ws.candidate_support,
                eps_denom=float(getattr(params, "eps_denom", 1e-12)),
            )
            score_support = _topk_score_keys(legacy_scores, max(0, int(getattr(params, "nadd", 0))))
            if int(getattr(params, "Kv", 0)) > 0:
                score_support |= topk_abs_keys_sparse(ws.v_proj, int(getattr(params, "Kv", 0)))
        else:
            score_support = _topk_score_keys(_score_from_vector(score_vec, used_selector), _tmp_keep(params))
            if bool(getattr(params, "include_current_support", True)):
                score_support |= q_k.support()

        v_temp = project_sparse_to_support(ws.v_proj, score_support)
        discarded_weight = float(np.sqrt(max(sparse_norm_sq(ws.v_proj) - sparse_norm_sq(v_temp), 0.0)))
        orth_residual = orthogonalize_against_basis(v_temp, basis)

        if bool(getattr(params, "preserve_prev_support", True)):
            preserve_cap = _runtime_preserve_prev_support_cap(params)
            if preserve_cap > 0:
                preserve_support = topk_abs_keys_sparse(q_k, preserve_cap)

        compressed = compress_vector(
            orth_residual,
            max(1, int(getattr(params, "nkeep", 1))),
            preserve_support=preserve_support if preserve_support else None,
        )
        q_next = normalize_sparse(compressed)

    matvec_error_norm: float | None = None
    relative_matvec_error: float | None = None
    if ws.exact_v is not None:
        diff = add_scaled_sparse(ws.exact_v, -1.0, ws.v_proj)
        matvec_error_norm = sparse_norm(diff)
        exact_norm = sparse_norm(ws.exact_v)
        relative_matvec_error = matvec_error_norm / max(exact_norm, 1e-30)
        hq_norm = exact_norm
    else:
        hq_norm = sparse_norm(ws.v_proj)

    diag = KrylovStepDiagnostics(
        q_norm=sparse_norm(q_k),
        hq_norm=hq_norm,
        projected_matvec_norm=sparse_norm(ws.v_proj),
        discarded_weight=discarded_weight,
        orthogonalized_residual_norm=sparse_norm(orth_residual),
        post_compression_residual_norm=sparse_norm(compressed),
        max_basis_overlap=_max_basis_overlap(q_next, basis),
        selector_mode_used=used_selector,
        candidate_mode_used=candidate_mode_from_params(params, default="selected_ci"),
        num_candidates_generated=len(ws.candidate_pool),
        num_tmp_kept=len(v_temp.coeffs),
        num_final_kept=len(compressed.coeffs),
        matvec_error_norm=matvec_error_norm,
        relative_matvec_error=relative_matvec_error,
        compression_policy_used=compression_policy,
        propagation_support_size=len(ws.propagation_support),
        propagation_dropped_norm=float(ws.propagation_dropped_norm),
        propagation_used_fallback=bool(ws.propagation_used_fallback),
        store_dropped_norm=store_dropped_norm,
        store_used_fallback=store_used_fallback,
        selector_reference_energy=selector_reference_energy,
        probe_residual_norm=probe_residual_norm,
        rebuilt_residual_norm=rebuilt_residual_norm,
        propagation_score_mode=propagation_score_mode,
        storage_score_mode=storage_score_mode,
        target_ritz_energy=target_ritz_energy,
        target_basis_size=target_basis_size,
        selector_score_mode=selector_score_mode,
        next_propagation_support_size=len(propagation_score_support) if propagation_score_support else None,
        delayed_propagation_used_fallback=bool(ws.propagation_used_fallback) if used_selector == "pt2_residual" else None,
    )
    artifacts = KrylovStepArtifacts(
        requested_selector=requested_selector,
        used_selector=used_selector,
        score_vector=score_vec,
        v_temp=v_temp,
        orthogonalized_residual=orth_residual,
        compressed=compressed,
        tmp_keep=_tmp_keep(params),
        preserve_cap=preserve_cap,
        preserve_support=preserve_support,
        candidate_pool=set(ws.candidate_pool),
        parent_support=set(ws.parent_support),
        propagation_score_support=propagation_score_support,
        score_support=score_support,
    )
    if used_selector == "pt2_residual":
        q_next.delayed_propagation_support = set(propagation_score_support)
    return q_next, diag, ws, artifacts


def build_next_krylov_vector(
    q_k: SparseCIVector,
    basis: list[SparseCIVector],
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SparseCIVector, KrylovStepDiagnostics, SparseProjectionWorkspace]:
    q_next, diag, ws, _ = analyze_krylov_step(
        q_k,
        basis,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        rng=rng,
    )
    return q_next, diag, ws
