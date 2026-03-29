"""Build the determinant space used to approximate ``H q_k``.

This module is intentionally more technical than the beginner examples. It is
responsible for deciding which determinants are explored, scored, and kept
during each sparse Krylov update.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .sparse_ci import (
    SparseCIVector,
    SparseProjectionWorkspace,
    compress_topk_sparse,
    full_hdiag_map,
    hdiag_map_from_array,
    normalize_sparse,
    project_sparse_to_support,
    sparse_from_dense,
    sparse_from_scivector,
    sparse_space_from_scivector,
    sparse_norm_sq,
    threshold_prune_sparse,
)

DetKey = tuple[int, int]
_THRESHOLD_MAX_CANDIDATES = 5_000_000


def _normalize_mode(raw: Any, default: str) -> str:
    value = default if raw is None else raw
    return str(value).strip().lower().replace("-", "_")


def candidate_mode_from_params(params: Any, *, default: str = "selected_ci") -> str:
    if hasattr(params, "candidate_mode") and getattr(params, "candidate_mode") is not None:
        mode = _normalize_mode(getattr(params, "candidate_mode"), default)
    else:
        mode = _normalize_mode(getattr(params, "selector_mode", default), default)

    aliases = {
        "selected_ci": "selected_ci",
        "selectedci": "selected_ci",
        "sci": "selected_ci",
        "cipsi": "selected_ci",
        "connected_sd": "connected_sd",
        "connected": "connected_sd",
        "generator_sd": "connected_sd",
        "stochastic": "stochastic",
        "full": "full",
        "dense": "full",
    }
    return aliases.get(mode, mode)


def selector_mode_from_params(params: Any, *, default: str = "abs_residual") -> str:
    mode = _normalize_mode(getattr(params, "selector_mode", default), default)
    aliases = {
        "selected_ci": "pt2_legacy",
        "selectedci": "pt2_legacy",
        "sci": "pt2_legacy",
        "cipsi": "pt2_legacy",
        "stochastic": "pt2_legacy",
        "full": "abs_hq",
        "dense": "abs_hq",
        "ritz_pt2": "ground_state_pt2",
    }
    return aliases.get(mode, mode)


def compression_policy_from_params(params: Any, *, default: str = "topk") -> str:
    mode = _normalize_mode(getattr(params, "compression_policy", default), default)
    aliases = {
        "topk": "topk",
        "legacy": "topk",
        "threshold": "threshold",
        "threshold_only": "threshold",
        "tol": "threshold",
        "fixed_budget": "fixed_budget",
        "fixedbudget": "fixed_budget",
        "budget": "fixed_budget",
    }
    return aliases.get(mode, mode)


def selector_depth_from_params(params: Any) -> int:
    return max(1, int(getattr(params, "selector_depth", 1)))


def _selector_cutoff_from_params(params: Any, name: str, default: float) -> float:
    return float(getattr(params, name, default))


def _generator_keep_from_params(params: Any, qk: SparseCIVector) -> int:
    raw = getattr(params, "generator_keep", len(qk.coeffs))
    return max(1, min(len(qk.coeffs), int(raw)))


def _preserve_current_support_from_params(params: Any) -> bool:
    return bool(getattr(params, "include_current_support", True))


def _threshold_abs_from_params(params: Any, name: str) -> float:
    return max(0.0, float(getattr(params, name, 0.0)))


def _fixed_budget_keep_from_params(params: Any, name: str, default: int) -> int:
    return max(1, int(getattr(params, name, default)))


def _propagation_source(
    qk: SparseCIVector,
    params: Any,
) -> tuple[SparseCIVector, set[DetKey], float, bool]:
    compression_policy = compression_policy_from_params(params, default="topk")
    if compression_policy == "fixed_budget":
        qprop = compress_topk_sparse(qk, _fixed_budget_keep_from_params(params, "fixed_propagate_keep", 10_000))
        dropped_norm_sq = max(0.0, sparse_norm_sq(qk) - sparse_norm_sq(qprop))
        return qprop, qprop.support(), float(np.sqrt(dropped_norm_sq)), False
    if compression_policy != "threshold":
        return qk.copy(), qk.support(), 0.0, False
    qprop, dropped_norm, used_fallback = threshold_prune_sparse(
        qk,
        _threshold_abs_from_params(params, "propagate_abs_tol"),
        keep_largest_if_empty=True,
    )
    return qprop, qprop.support(), dropped_norm, used_fallback


def _rectangularize_support(support: set[DetKey]) -> set[DetKey]:
    if not support:
        return set()
    strsa = {int(stra) for stra, _ in support}
    strsb = {int(strb) for _, strb in support}
    return {(stra, strb) for stra in strsa for strb in strsb}


def _select_parent_support(qk: SparseCIVector, params: Any) -> set[DetKey]:
    if not qk.coeffs:
        return set()

    cutoff = float(getattr(params, "generator_abs_coeff_cutoff", 0.0))
    ranked = sorted(qk.coeffs.items(), key=lambda item: abs(item[1]), reverse=True)
    filtered = [(key, val) for key, val in ranked if abs(val) >= cutoff]
    if not filtered:
        filtered = ranked[:1]

    nparent = min(len(filtered), _generator_keep_from_params(params, qk))
    max_parents = getattr(params, "max_parents_for_expansion", None)
    if max_parents is not None:
        nparent = min(nparent, max(1, int(max_parents)))
    nparent = max(1, nparent)
    return {key for key, _ in filtered[:nparent]}


def _bit_to_occ(bitstr: int, norb: int) -> list[int]:
    out: list[int] = []
    x = int(bitstr)
    while x:
        lsb = x & -x
        p = lsb.bit_length() - 1
        if p < norb:
            out.append(int(p))
        x ^= lsb
    return out


def _bit_to_vir(bitstr: int, norb: int) -> list[int]:
    full = (1 << norb) - 1
    vir = full ^ int(bitstr)
    return _bit_to_occ(vir, norb)


def _choose_two_distinct(rng: np.random.Generator, items: list[int]) -> tuple[int, int]:
    i = int(rng.integers(0, len(items)))
    j = int(rng.integers(0, len(items) - 1))
    if j >= i:
        j += 1
    a = items[i]
    b = items[j]
    return (a, b) if a < b else (b, a)


def _spawn_once(
    rng: np.random.Generator,
    *,
    occ_a: list[int],
    vir_a: list[int],
    occ_b: list[int],
    vir_b: list[int],
    p_double: float,
    mixed_double_weight: float,
) -> tuple[list[int], list[int], list[int], list[int], float]:
    nocc_a = len(occ_a)
    nvir_a = len(vir_a)
    nocc_b = len(occ_b)
    nvir_b = len(vir_b)

    n_single_a = nocc_a * nvir_a
    n_single_b = nocc_b * nvir_b
    n_double_aa = (nocc_a * (nocc_a - 1) // 2) * (nvir_a * (nvir_a - 1) // 2)
    n_double_bb = (nocc_b * (nocc_b - 1) // 2) * (nvir_b * (nvir_b - 1) // 2)
    n_double_ab = nocc_a * nvir_a * nocc_b * nvir_b

    do_double = rng.random() < p_double
    if do_double and (n_double_aa + n_double_bb + n_double_ab) == 0:
        do_double = False
    if (not do_double) and (n_single_a + n_single_b) == 0:
        do_double = True

    if not do_double:
        tot = n_single_a + n_single_b
        pa = (n_single_a / tot) if tot > 0 else 0.0
        if rng.random() < pa:
            hole = occ_a[int(rng.integers(0, nocc_a))]
            part = vir_a[int(rng.integers(0, nvir_a))]
            p_gen = (1.0 - p_double) * pa * (1.0 / nocc_a) * (1.0 / nvir_a)
            return [hole], [part], [], [], float(p_gen)

        hole = occ_b[int(rng.integers(0, nocc_b))]
        part = vir_b[int(rng.integers(0, nvir_b))]
        p_gen = (1.0 - p_double) * (1.0 - pa) * (1.0 / nocc_b) * (1.0 / nvir_b)
        return [], [], [hole], [part], float(p_gen)

    w_aa = float(n_double_aa)
    w_bb = float(n_double_bb)
    w_ab = float(n_double_ab) * float(mixed_double_weight)
    wtot = w_aa + w_bb + w_ab
    if wtot <= 0.0:
        return [], [], [], [], 0.0

    r = rng.random() * wtot
    if r < w_aa:
        i, j = _choose_two_distinct(rng, occ_a)
        a, b = _choose_two_distinct(rng, vir_a)
        p_gen = (
            p_double
            * (w_aa / wtot)
            * (1.0 / (nocc_a * (nocc_a - 1) / 2.0))
            * (1.0 / (nvir_a * (nvir_a - 1) / 2.0))
        )
        return [i, j], [a, b], [], [], float(p_gen)

    if r < (w_aa + w_bb):
        i, j = _choose_two_distinct(rng, occ_b)
        a, b = _choose_two_distinct(rng, vir_b)
        p_gen = (
            p_double
            * (w_bb / wtot)
            * (1.0 / (nocc_b * (nocc_b - 1) / 2.0))
            * (1.0 / (nvir_b * (nvir_b - 1) / 2.0))
        )
        return [], [], [i, j], [a, b], float(p_gen)

    hole_a = occ_a[int(rng.integers(0, nocc_a))]
    part_a = vir_a[int(rng.integers(0, nvir_a))]
    hole_b = occ_b[int(rng.integers(0, nocc_b))]
    part_b = vir_b[int(rng.integers(0, nvir_b))]
    p_gen = p_double * (w_ab / wtot) * (1.0 / nocc_a) * (1.0 / nvir_a) * (1.0 / nocc_b) * (1.0 / nvir_b)
    return [hole_a], [part_a], [hole_b], [part_b], float(p_gen)


def discover_candidates_support_stochastic(
    *,
    norb: int,
    nelec: tuple[int, int],
    qk: SparseCIVector,
    params: Any,
    rng: np.random.Generator,
) -> set[DetKey]:
    if not qk.coeffs or int(getattr(params, "n_walkers", 0)) <= 0:
        return set()

    parents = list(qk.coeffs.items())
    qvals = np.asarray([abs(val) for _, val in parents], dtype=float)
    parent_power = float(getattr(params, "parent_power", 1.0))
    if parent_power != 1.0:
        qvals = qvals**parent_power
    tot = float(np.sum(qvals))
    if tot <= 0.0:
        return set()
    p_parent = qvals / tot

    cand: set[DetKey] = set()
    nwalk = int(getattr(params, "n_walkers", 0))
    p_double = float(getattr(params, "p_double", 0.6))
    mixed_double_weight = float(getattr(params, "mixed_double_weight", 1.0))
    verbose = bool(getattr(params, "verbose", False))

    parent_cache: dict[DetKey, tuple[list[int], list[int], list[int], list[int]]] = {}

    for w in range(nwalk):
        if verbose and nwalk >= 20000 and (w % 5000) == 0 and w > 0:
            print(f"      [SELECTOR-STOCHASTIC] walkers: {w}/{nwalk}")

        t = int(rng.choice(len(parents), p=p_parent))
        (stra, strb), _ = parents[t]

        cached = parent_cache.get((stra, strb))
        if cached is None:
            occ_a = _bit_to_occ(stra, norb)
            occ_b = _bit_to_occ(strb, norb)
            vir_a = _bit_to_vir(stra, norb)
            vir_b = _bit_to_vir(strb, norb)
            parent_cache[(stra, strb)] = (occ_a, vir_a, occ_b, vir_b)
        else:
            occ_a, vir_a, occ_b, vir_b = cached

        holes_a, parts_a, holes_b, parts_b, _ = _spawn_once(
            rng,
            occ_a=occ_a,
            vir_a=vir_a,
            occ_b=occ_b,
            vir_b=vir_b,
            p_double=p_double,
            mixed_double_weight=mixed_double_weight,
        )
        if len(holes_a) + len(holes_b) == 0:
            continue

        stra2 = int(stra)
        for h in holes_a:
            stra2 ^= 1 << int(h)
        for p in parts_a:
            stra2 ^= 1 << int(p)

        strb2 = int(strb)
        for h in holes_b:
            strb2 ^= 1 << int(h)
        for p in parts_b:
            strb2 ^= 1 << int(p)

        cand.add((int(stra2), int(strb2)))

    return cand


def _normalize_sci_vector(ci_sci):
    from pyscf.fci import selected_ci

    arr = np.asarray(ci_sci, dtype=float)
    nrm = np.linalg.norm(arr.ravel())
    if nrm <= 0.0:
        return selected_ci._as_SCIvector(arr.copy(), ci_sci._strs)
    return selected_ci._as_SCIvector(arr / nrm, ci_sci._strs)


def _enumerate_connected_sd_support(parents: set[DetKey], norb: int) -> set[DetKey]:
    cand: set[DetKey] = set()
    for stra, strb in parents:
        occ_a = _bit_to_occ(stra, norb)
        occ_b = _bit_to_occ(strb, norb)
        vir_a = _bit_to_vir(stra, norb)
        vir_b = _bit_to_vir(strb, norb)

        cand.add((int(stra), int(strb)))

        for i in occ_a:
            for a in vir_a:
                cand.add((int(stra ^ (1 << i) ^ (1 << a)), int(strb)))
        for i in occ_b:
            for a in vir_b:
                cand.add((int(stra), int(strb ^ (1 << i) ^ (1 << a))))

        for (i, j) in combinations(occ_a, 2):
            for (a, b) in combinations(vir_a, 2):
                cand.add((int(stra ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b)), int(strb)))
        for (i, j) in combinations(occ_b, 2):
            for (a, b) in combinations(vir_b, 2):
                cand.add((int(stra), int(strb ^ (1 << i) ^ (1 << j) ^ (1 << a) ^ (1 << b))))
        for i in occ_a:
            for a in vir_a:
                stra2 = int(stra ^ (1 << i) ^ (1 << a))
                for j in occ_b:
                    for b in vir_b:
                        cand.add((stra2, int(strb ^ (1 << j) ^ (1 << b))))
    return cand


def _make_selected_ci_solver(params: Any):
    from pyscf.fci import selected_ci

    myci = selected_ci.SelectedCI()
    myci.ci_coeff_cutoff = _selector_cutoff_from_params(params, "selector_ci_coeff_cutoff", 5e-4)
    myci.select_cutoff = _selector_cutoff_from_params(params, "selector_cutoff", 5e-4)
    return myci


def build_candidate_pool_impl(
    *,
    qk: SparseCIVector,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[set[DetKey], set[DetKey]]:
    from pyscf.fci import selected_ci

    compression_policy = compression_policy_from_params(params, default="topk")
    candidate_mode = candidate_mode_from_params(params, default="selected_ci")
    qprop, current_support, _, _ = _propagation_source(qk, params)
    if compression_policy in {"threshold", "fixed_budget"}:
        parent_support = set(current_support)
        parent_vec = qprop.copy()
    else:
        parent_support = _select_parent_support(qk, params)
        parent_vec = normalize_sparse(project_sparse_to_support(qk, parent_support))
        if not parent_vec.coeffs:
            parent_vec = normalize_sparse(qk)
            parent_support = parent_vec.support()

    include_current_support = _preserve_current_support_from_params(params)

    if compression_policy == "threshold":
        if candidate_mode == "full":
            if hdiag_full is None:
                raise ValueError("full candidate mode requires hdiag_full")
            return parent_support, set(hdiag_full)
        candidate_pool = _enumerate_connected_sd_support(parent_support, norb) | current_support
        if len(candidate_pool) > _THRESHOLD_MAX_CANDIDATES:
            raise RuntimeError(
                "threshold compression policy exceeded internal candidate limit "
                f"({_THRESHOLD_MAX_CANDIDATES}) while building connected-SD candidates; "
                "loosen thresholds or switch back to topk compression."
        )
        return parent_support, candidate_pool

    if compression_policy == "fixed_budget" and candidate_mode == "full":
        if hdiag_full is None:
            raise ValueError("full candidate mode requires hdiag_full")
        return parent_support, set(hdiag_full)

    if candidate_mode == "full":
        if hdiag_full is None:
            raise ValueError("full candidate mode requires hdiag_full")
        candidate_pool = set(hdiag_full)
        if include_current_support:
            candidate_pool |= current_support
        return parent_support, candidate_pool

    depth = selector_depth_from_params(params)

    if candidate_mode == "connected_sd":
        candidate_pool = _enumerate_connected_sd_support(parent_support, norb)
        if include_current_support:
            candidate_pool |= current_support
        return parent_support, candidate_pool

    if candidate_mode == "selected_ci":
        myci = _make_selected_ci_solver(params)
        work = _normalize_sci_vector(parent_vec.to_scivector())
        expanded = work
        for d in range(depth):
            expanded = myci.enlarge_space(work, eri8, norb, nelec)
            if d + 1 < depth:
                v_sci = selected_ci.contract_2e(h2eff, expanded, norb, nelec)
                work = _normalize_sci_vector(v_sci)
        candidate_pool = set(sparse_space_from_scivector(expanded).keys())
        if include_current_support:
            candidate_pool |= current_support
        return parent_support, _rectangularize_support(candidate_pool)

    if candidate_mode == "stochastic":
        if rng is None:
            raise ValueError("stochastic candidate mode requires an RNG")
        selected_support = set(current_support if include_current_support else parent_support)
        work_vec = parent_vec
        accumulated_candidates: set[DetKey] = set()

        for d in range(depth):
            new_candidates = discover_candidates_support_stochastic(
                norb=norb,
                nelec=nelec,
                qk=work_vec,
                params=params,
                rng=rng,
            )
            accumulated_candidates |= new_candidates
            selected_support |= new_candidates
            if d + 1 < depth:
                work_sci = work_vec.to_scivector_on_support(selected_support)
                v_sci = selected_ci.contract_2e(h2eff, work_sci, norb, nelec)
                work_vec = normalize_sparse(sparse_from_scivector(v_sci))
                if not work_vec.coeffs:
                    break

        return parent_support, _rectangularize_support(selected_support | accumulated_candidates)

    raise ValueError(
        f"Unknown candidate_mode {candidate_mode!r}. Expected one of: connected_sd, selected_ci, stochastic, full."
    )


def compute_projected_matvec_impl(
    *,
    qk: SparseCIVector,
    candidate_pool: set[DetKey],
    parent_support: set[DetKey],
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
) -> SparseProjectionWorkspace:
    from pyscf.fci import direct_spin1, selected_ci

    compression_policy = compression_policy_from_params(params, default="topk")
    candidate_mode = candidate_mode_from_params(params, default="selected_ci")
    qprop, prop_support, prop_dropped_norm, prop_used_fallback = _propagation_source(qk, params)
    exact_v: SparseCIVector | None = None

    if bool(getattr(params, "exact_hq_diagnostics", False)):
        dense_q = qprop.to_dense(norb=norb, nelec=nelec)
        dense_v = np.asarray(direct_spin1.contract_2e(h2eff, dense_q, norb, nelec), dtype=float)
        exact_v = sparse_from_dense(dense_v, norb=norb, nelec=nelec)

    if candidate_mode == "full":
        if hdiag_full is None:
            raise ValueError("full candidate mode requires hdiag_full")
        dense_q = qprop.to_dense(norb=norb, nelec=nelec)
        dense_v = np.asarray(direct_spin1.contract_2e(h2eff, dense_q, norb, nelec), dtype=float)
        v_proj = sparse_from_dense(dense_v, norb=norb, nelec=nelec)
        full_support = set(hdiag_full)
        return SparseProjectionWorkspace(
            v_proj=v_proj,
            hdiag=hdiag_full,
            space=sparse_space_from_scivector(v_proj.to_scivector() if v_proj.coeffs else qk.to_scivector()),
            candidate_support=full_support - prop_support,
            candidate_pool=full_support,
            parent_support=set(parent_support),
            propagation_support=set(prop_support),
            propagation_dropped_norm=prop_dropped_norm,
            propagation_used_fallback=prop_used_fallback,
            exact_v=exact_v if exact_v is not None else v_proj,
        )

    support = _rectangularize_support(set(candidate_pool) | prop_support)
    if compression_policy == "threshold" and len(support) > _THRESHOLD_MAX_CANDIDATES:
        raise RuntimeError(
            "threshold compression policy exceeded internal support limit "
            f"({_THRESHOLD_MAX_CANDIDATES}) while building the projected Hq workspace; "
            "loosen thresholds or switch back to topk compression."
        )
    work_sci = qprop.to_scivector_on_support(support)
    v_sci = selected_ci.contract_2e(h2eff, work_sci, norb, nelec)
    space = sparse_space_from_scivector(v_sci)
    actual_pool = set(space.keys())
    hdiag_sel = selected_ci.make_hdiag(h1, eri8, work_sci._strs, norb, nelec).reshape(work_sci.shape)
    hdiag = hdiag_map_from_array(hdiag_sel, space=space)
    v_proj = sparse_from_scivector(v_sci)
    return SparseProjectionWorkspace(
        v_proj=v_proj,
        hdiag=hdiag,
        space=space,
        candidate_support=actual_pool - prop_support,
        candidate_pool=actual_pool,
        parent_support=set(parent_support),
        propagation_support=set(prop_support),
        propagation_dropped_norm=prop_dropped_norm,
        propagation_used_fallback=prop_used_fallback,
        exact_v=exact_v,
    )


def build_projection_workspace(
    *,
    qk: SparseCIVector,
    h1: np.ndarray,
    eri8: np.ndarray,
    h2eff: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    params: Any,
    hdiag_full: dict[DetKey, float] | None = None,
    rng: np.random.Generator | None = None,
) -> SparseProjectionWorkspace:
    parent_support, candidate_pool = build_candidate_pool_impl(
        qk=qk,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=params,
        hdiag_full=hdiag_full,
        rng=rng,
    )
    return compute_projected_matvec_impl(
        qk=qk,
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


def make_full_hdiag_map(
    *,
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> dict[DetKey, float]:
    from pyscf.fci import cistring, direct_spin1

    neleca, nelecb = nelec
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    hdiag_full = direct_spin1.make_hdiag(h1, eri8, norb, nelec)
    hdiag_full = np.asarray(hdiag_full, dtype=float).reshape(na, nb)
    return full_hdiag_map(hdiag_full=hdiag_full, norb=norb, nelec=nelec)
