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
from .sparse_ci import (
    SparseCIVector,
    SparseProjectionWorkspace,
    add_scaled_sparse,
    compress_topk_sparse,
    compress_with_support_sparse,
    inner_sparse,
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


@dataclass(frozen=True)
class KrylovStepArtifacts:
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
    return compression_policy_from_params(params, default="topk")


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


def _score_from_vector(vec: SparseCIVector, mode: str) -> dict[DetKey, float]:
    if mode in {"abs_hq", "abs_residual"}:
        return {key: abs(val) for key, val in vec.coeffs.items() if abs(val) > 0.0}
    if mode in {"sq_hq", "sq_residual"}:
        return {key: abs(val) ** 2 for key, val in vec.coeffs.items() if abs(val) > 0.0}
    raise ValueError(f"Unsupported score mode {mode!r}")


def _max_basis_overlap(vec: SparseCIVector, basis: list[SparseCIVector]) -> float:
    if not vec.coeffs or not basis:
        return 0.0
    return max(abs(inner_sparse(vec, q)) for q in basis)


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

    requested_selector = selector_mode_from_params(params, default="abs_residual")
    used_selector = requested_selector
    score_vec = ws.v_proj
    compression_policy = _compression_policy(params)

    if requested_selector in {"abs_residual", "sq_residual"}:
        if basis:
            score_vec = orthogonalize_against_basis(ws.v_proj, basis)
        else:
            used_selector = "abs_hq" if requested_selector == "abs_residual" else "sq_hq"
            score_vec = ws.v_proj

    score_support: set[DetKey] = set()
    preserve_support: set[DetKey] = set()
    preserve_cap = 0
    store_dropped_norm = 0.0
    store_used_fallback = False

    if compression_policy == "threshold":
        v_temp = ws.v_proj.copy()
        discarded_weight = 0.0
        orth_residual = orthogonalize_against_basis(v_temp, basis)
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
        v_temp = ws.v_proj.copy()
        discarded_weight = 0.0
        orth_residual = orthogonalize_against_basis(v_temp, basis)
        compressed = compress_topk_sparse(orth_residual, _fixed_store_keep(params))
        store_dropped_norm = float(np.sqrt(max(sparse_norm_sq(orth_residual) - sparse_norm_sq(compressed), 0.0)))
        q_next = normalize_sparse(compressed)
        score_support = set(score_vec.coeffs)
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
        candidate_mode_used=(
            "full"
            if compression_policy in {"threshold", "fixed_budget"} and candidate_mode_from_params(params, default="selected_ci") == "full"
            else (
                "connected_sd"
                if compression_policy in {"threshold", "fixed_budget"}
                else candidate_mode_from_params(params, default="selected_ci")
            )
        ),
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
        score_support=score_support,
    )
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
