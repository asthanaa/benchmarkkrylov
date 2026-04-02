"""Sparse determinant-vector data structures used by the CCIK solvers.

The objects here are the main reason the solver can avoid keeping a full dense
CI vector in memory for every Krylov step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

DetKey = tuple[int, int]
_ZERO_TOL = 1e-15


def _det_sort_key(key: DetKey) -> tuple[int, int]:
    return (int(key[0]), int(key[1]))


def _canonical_items(
    coeffs: Mapping[DetKey, float],
    *,
    zero_tol: float = _ZERO_TOL,
    min_abs: float = 0.0,
) -> list[tuple[DetKey, float]]:
    tau = max(float(zero_tol), float(min_abs))
    return [
        ((int(key[0]), int(key[1])), float(val))
        for key, val in sorted(coeffs.items(), key=lambda item: _det_sort_key(item[0]))
        if abs(float(val)) > tau
    ]


def _canonicalize_coeffs(coeffs: Mapping[DetKey, float], *, zero_tol: float = _ZERO_TOL) -> dict[DetKey, float]:
    return {key: val for key, val in _canonical_items(coeffs, zero_tol=zero_tol)}


def _rank_items_by_abs_qp(items: list[tuple[DetKey, float]]) -> list[tuple[DetKey, float]]:
    ranked = list(items)
    # QP ranks magnitudes after determinant-canonicalization. Python's stable sort
    # then preserves the canonical determinant order within equal-|coef| groups.
    ranked.sort(key=lambda item: abs(item[1]))
    return ranked


def _rank_items_by_abs_qp_grouped(
    items: list[tuple[DetKey, float]],
    *,
    rel_tol: float = 1.0e-12,
    abs_tol: float = 1.0e-15,
) -> list[tuple[DetKey, float]]:
    ranked = _rank_items_by_abs_qp(items)
    if len(ranked) <= 1:
        return ranked

    grouped = list(ranked)
    left = 0
    while left < len(grouped):
        ref_abs = abs(float(grouped[left][1]))
        tie_tol = max(float(abs_tol), float(rel_tol) * max(1.0, ref_abs))
        right = left
        while right + 1 < len(grouped):
            if abs(abs(float(grouped[right + 1][1])) - ref_abs) > tie_tol:
                break
            right += 1
        if right > left:
            grouped[left : right + 1] = sorted(grouped[left : right + 1], key=lambda item: _det_sort_key(item[0]))
        left = right + 1
    return grouped


@dataclass
class SparseCIVector:
    """A sparse CI vector keyed by determinant pairs.

    Plain-language purpose:
    - store only the determinant amplitudes that are currently being kept

    Chemistry meaning:
    - represents a compressed CI wavefunction in the determinant basis
    """

    coeffs: dict[DetKey, float]
    delayed_propagation_support: set[DetKey] = field(default_factory=set)

    def copy(self) -> "SparseCIVector":
        return SparseCIVector(
            dict(self.coeffs),
            delayed_propagation_support=set(self.delayed_propagation_support),
        )

    def nnz(self) -> int:
        return len(self.coeffs)

    def support(self) -> set[DetKey]:
        return set(self.coeffs)

    def get(self, key: DetKey, default: float = 0.0) -> float:
        return float(self.coeffs.get(key, default))

    def to_dense(self, *, norb: int, nelec: tuple[int, int]) -> np.ndarray:
        from pyscf.fci import cistring

        neleca, nelecb = nelec
        na = cistring.num_strings(norb, neleca)
        nb = cistring.num_strings(norb, nelecb)
        out = np.zeros((na, nb), dtype=float)
        for (stra, strb), coeff in self.coeffs.items():
            ia = int(cistring.str2addr(norb, neleca, int(stra)))
            ib = int(cistring.str2addr(norb, nelecb, int(strb)))
            out[ia, ib] = float(coeff)
        return out

    def to_scivector(self):
        from pyscf.fci import selected_ci

        if not self.coeffs:
            raise ValueError("cannot convert an empty sparse CI vector to SelectedCI format")

        strsa = np.asarray(sorted({int(stra) for stra, _ in self.coeffs}), dtype=np.int64)
        strsb = np.asarray(sorted({int(strb) for _, strb in self.coeffs}), dtype=np.int64)
        map_a = {int(stra): idx for idx, stra in enumerate(strsa.tolist())}
        map_b = {int(strb): idx for idx, strb in enumerate(strsb.tolist())}
        coeff = np.zeros((len(strsa), len(strsb)), dtype=float)
        for (stra, strb), val in self.coeffs.items():
            coeff[map_a[int(stra)], map_b[int(strb)]] = float(val)
        return selected_ci._as_SCIvector(coeff, (strsa, strsb))

    def to_scivector_on_support(self, support: Iterable[DetKey]):
        from pyscf.fci import selected_ci

        support_set = {(int(stra), int(strb)) for stra, strb in support}
        support_set.update((int(stra), int(strb)) for stra, strb in self.coeffs)
        if not support_set:
            raise ValueError("cannot convert an empty support to SelectedCI format")

        strsa = np.asarray(sorted({int(stra) for stra, _ in support_set}), dtype=np.int64)
        strsb = np.asarray(sorted({int(strb) for _, strb in support_set}), dtype=np.int64)
        map_a = {int(stra): idx for idx, stra in enumerate(strsa.tolist())}
        map_b = {int(strb): idx for idx, strb in enumerate(strsb.tolist())}
        coeff = np.zeros((len(strsa), len(strsb)), dtype=float)
        for (stra, strb), val in self.coeffs.items():
            coeff[map_a[int(stra)], map_b[int(strb)]] = float(val)
        return selected_ci._as_SCIvector(coeff, (strsa, strsb))


@dataclass(frozen=True)
class SparseDeterminantSpace:
    """A rectangular determinant space built from alpha and beta string lists."""

    strsa: tuple[int, ...]
    strsb: tuple[int, ...]

    def keys(self) -> list[DetKey]:
        return [(int(stra), int(strb)) for stra in self.strsa for strb in self.strsb]

    def size(self) -> int:
        return len(self.strsa) * len(self.strsb)


@dataclass(frozen=True)
class SparseProjectionWorkspace:
    """Bundle the projected ``H q_k`` data produced during one Krylov step."""

    v_proj: SparseCIVector
    hdiag: dict[DetKey, float]
    space: SparseDeterminantSpace
    candidate_support: set[DetKey]
    candidate_pool: set[DetKey] = field(default_factory=set)
    parent_support: set[DetKey] = field(default_factory=set)
    propagation_support: set[DetKey] = field(default_factory=set)
    propagation_dropped_norm: float = 0.0
    propagation_used_fallback: bool = False
    exact_v: SparseCIVector | None = None



def _clean_coeffs(coeffs: Mapping[DetKey, float], *, zero_tol: float = _ZERO_TOL) -> dict[DetKey, float]:
    return _canonicalize_coeffs(coeffs, zero_tol=zero_tol)



def sparse_from_mapping(coeffs: Mapping[DetKey, float], *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    return SparseCIVector(_clean_coeffs(coeffs, zero_tol=zero_tol))



def sparse_from_dense(ci: np.ndarray, *, norb: int, nelec: tuple[int, int], zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    from pyscf.fci import cistring

    neleca, nelecb = nelec
    nz = np.argwhere(np.abs(ci) > zero_tol)
    coeffs: dict[DetKey, float] = {}
    for ia, ib in nz:
        stra = int(cistring.addr2str(norb, neleca, int(ia)))
        strb = int(cistring.addr2str(norb, nelecb, int(ib)))
        coeffs[(stra, strb)] = float(ci[int(ia), int(ib)])
    return SparseCIVector(coeffs)



def sparse_from_scivector(ci_sci, *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    arr = np.asarray(ci_sci, dtype=float)
    strsa, strsb = ci_sci._strs
    nz = np.argwhere(np.abs(arr) > zero_tol)
    coeffs: dict[DetKey, float] = {}
    for ia, ib in nz:
        coeffs[(int(strsa[int(ia)]), int(strsb[int(ib)]))] = float(arr[int(ia), int(ib)])
    return SparseCIVector(coeffs)



def sparse_space_from_scivector(ci_sci) -> SparseDeterminantSpace:
    strsa, strsb = ci_sci._strs
    return SparseDeterminantSpace(tuple(int(x) for x in strsa.tolist()), tuple(int(x) for x in strsb.tolist()))



def hdiag_map_from_array(hdiag: np.ndarray, *, space: SparseDeterminantSpace) -> dict[DetKey, float]:
    out: dict[DetKey, float] = {}
    arr = np.asarray(hdiag, dtype=float)
    for ia, stra in enumerate(space.strsa):
        for ib, strb in enumerate(space.strsb):
            out[(int(stra), int(strb))] = float(arr[ia, ib])
    return out



def sparse_norm_sq(vec: SparseCIVector) -> float:
    return float(sum(abs(val) ** 2 for val in vec.coeffs.values()))



def sparse_norm(vec: SparseCIVector) -> float:
    return float(np.sqrt(sparse_norm_sq(vec)))



def normalize_sparse(vec: SparseCIVector, *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    nrm = sparse_norm(vec)
    if nrm <= 0.0:
        return SparseCIVector({})
    return SparseCIVector(
        _canonicalize_coeffs({key: float(val / nrm) for key, val in vec.coeffs.items()}, zero_tol=zero_tol)
    )



def inner_sparse(vec1: SparseCIVector, vec2: SparseCIVector) -> float:
    if len(vec1.coeffs) <= len(vec2.coeffs):
        small = vec1.coeffs
        other = vec2.coeffs
    else:
        small = vec2.coeffs
        other = vec1.coeffs
    return float(sum(val * other.get(key, 0.0) for key, val in small.items()))



def add_scaled_sparse(base: SparseCIVector, alpha: float, other: SparseCIVector, *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    if abs(alpha) <= zero_tol or not other.coeffs:
        return base.copy()

    out = dict(base.coeffs)
    for key, val in other.coeffs.items():
        new_val = out.get(key, 0.0) + float(alpha) * float(val)
        if abs(new_val) > zero_tol:
            out[key] = new_val
        elif key in out:
            del out[key]
    return SparseCIVector(_canonicalize_coeffs(out, zero_tol=zero_tol))



def linear_combination_sparse(terms: Iterable[tuple[float, SparseCIVector]], *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    out: dict[DetKey, float] = {}
    for coeff, vec in terms:
        if abs(coeff) <= zero_tol:
            continue
        for key, val in vec.coeffs.items():
            out[key] = out.get(key, 0.0) + float(coeff) * float(val)
    return SparseCIVector(_canonicalize_coeffs(out, zero_tol=zero_tol))



def project_sparse_to_support(vec: SparseCIVector, support: set[DetKey], *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    if not support or not vec.coeffs:
        return SparseCIVector({})
    return SparseCIVector(
        _canonicalize_coeffs({key: val for key, val in vec.coeffs.items() if key in support}, zero_tol=zero_tol)
    )



def compress_topk_sparse(vec: SparseCIVector, nkeep: int | None = None, *, min_abs: float = 0.0) -> SparseCIVector:
    if nkeep is None or nkeep >= len(vec.coeffs):
        return SparseCIVector(_canonicalize_coeffs(vec.coeffs, zero_tol=max(_ZERO_TOL, float(min_abs))))
    eligible = _canonical_items(vec.coeffs, min_abs=min_abs)
    if not eligible:
        if not vec.coeffs:
            return SparseCIVector({})
        key, val = _rank_items_by_abs_qp(_canonical_items(vec.coeffs))[-1]
        return SparseCIVector(_canonicalize_coeffs({key: val}))
    ranked = _rank_items_by_abs_qp(eligible)
    kept = ranked[-int(nkeep) :]
    return SparseCIVector(_canonicalize_coeffs({key: val for key, val in kept}))


def compress_topk_sparse_grouped(
    vec: SparseCIVector,
    nkeep: int | None = None,
    *,
    min_abs: float = 0.0,
    rel_tol: float = 1.0e-12,
    abs_tol: float = 1.0e-15,
) -> SparseCIVector:
    if nkeep is None or nkeep >= len(vec.coeffs):
        return SparseCIVector(_canonicalize_coeffs(vec.coeffs, zero_tol=max(_ZERO_TOL, float(min_abs))))
    eligible = _canonical_items(vec.coeffs, min_abs=min_abs)
    if not eligible:
        if not vec.coeffs:
            return SparseCIVector({})
        key, val = _rank_items_by_abs_qp_grouped(_canonical_items(vec.coeffs), rel_tol=rel_tol, abs_tol=abs_tol)[-1]
        return SparseCIVector(_canonicalize_coeffs({key: val}))
    ranked = _rank_items_by_abs_qp_grouped(eligible, rel_tol=rel_tol, abs_tol=abs_tol)
    kept = ranked[-int(nkeep) :]
    return SparseCIVector(_canonicalize_coeffs({key: val for key, val in kept}))



def compress_with_support_sparse(
    vec: SparseCIVector,
    nkeep: int | None,
    *,
    preserve_support: Iterable[DetKey] | None = None,
    min_abs: float = 0.0,
) -> SparseCIVector:
    if nkeep is None or nkeep >= len(vec.coeffs):
        return SparseCIVector(_canonicalize_coeffs(vec.coeffs, zero_tol=max(_ZERO_TOL, float(min_abs))))
    if not vec.coeffs:
        return SparseCIVector({})

    preserve = set(preserve_support or ())
    kept: list[DetKey] = []
    if preserve:
        preserved = _rank_items_by_abs_qp(
            _canonical_items({key: vec.coeffs[key] for key in preserve if key in vec.coeffs}, min_abs=min_abs)
        )
        kept.extend(key for key, _ in preserved[-int(nkeep) :])

    if len(kept) < int(nkeep):
        kept_set = set(kept)
        remaining = _rank_items_by_abs_qp(
            _canonical_items({key: val for key, val in vec.coeffs.items() if key not in kept_set}, min_abs=min_abs)
        )
        kept.extend(key for key, _ in remaining[-(int(nkeep) - len(kept)) :])

    kept_set = set(kept)
    return SparseCIVector(_canonicalize_coeffs({key: val for key, val in vec.coeffs.items() if key in kept_set}))


def threshold_prune_sparse(
    vec: SparseCIVector,
    abs_tol: float,
    *,
    keep_largest_if_empty: bool = True,
) -> tuple[SparseCIVector, float, bool]:
    if not vec.coeffs:
        return SparseCIVector({}), 0.0, False

    tau = max(0.0, float(abs_tol))
    kept = _canonicalize_coeffs({key: float(val) for key, val in vec.coeffs.items() if abs(float(val)) >= tau})
    if kept:
        dropped_norm_sq = max(sparse_norm_sq(vec) - sparse_norm_sq(SparseCIVector(kept)), 0.0)
        return SparseCIVector(kept), float(np.sqrt(dropped_norm_sq)), False

    if not keep_largest_if_empty:
        return SparseCIVector({}), sparse_norm(vec), False

    key, val = _rank_items_by_abs_qp(_canonical_items(vec.coeffs))[-1]
    fallback = SparseCIVector(_canonicalize_coeffs({key: val}))
    dropped_norm_sq = max(sparse_norm_sq(vec) - sparse_norm_sq(fallback), 0.0)
    return fallback, float(np.sqrt(dropped_norm_sq)), True


def compress_threshold_sparse(
    vec: SparseCIVector,
    abs_tol: float,
    *,
    keep_largest_if_empty: bool = True,
) -> SparseCIVector:
    out, _, _ = threshold_prune_sparse(vec, abs_tol, keep_largest_if_empty=keep_largest_if_empty)
    return out


def topk_abs_keys_sparse(vec: SparseCIVector, k: int) -> set[DetKey]:
    if k <= 0 or not vec.coeffs:
        return set()
    ranked = _rank_items_by_abs_qp(_canonical_items(vec.coeffs))
    return {key for key, val in ranked[-int(k) :] if abs(float(val)) > 0.0}



def topk_positive_keys(values: Mapping[DetKey, float], k: int) -> set[DetKey]:
    if k <= 0:
        return set()
    ranked = [((int(key[0]), int(key[1])), float(val)) for key, val in sorted(values.items(), key=lambda item: _det_sort_key(item[0])) if float(val) > 0.0]
    ranked.sort(key=lambda item: item[1])
    return {key for key, _ in ranked[-int(k) :]}



def orthogonalize_sparse(vec: SparseCIVector, basis: list[SparseCIVector], *, zero_tol: float = _ZERO_TOL) -> SparseCIVector:
    r = vec.copy()
    for q in basis:
        coeff = inner_sparse(q, r)
        if abs(coeff) > zero_tol:
            r = add_scaled_sparse(r, -coeff, q, zero_tol=zero_tol)
    return r



def union_support(vectors: Iterable[SparseCIVector]) -> set[DetKey]:
    out: set[DetKey] = set()
    for vec in vectors:
        out.update(vec.coeffs)
    return out



def make_hf_sparse_vector(*, norb: int, nelec: tuple[int, int]) -> SparseCIVector:
    from pyscf.fci import cistring

    neleca, nelecb = nelec
    occ_a = 0
    occ_b = 0
    for p in range(neleca):
        occ_a |= 1 << p
    for p in range(nelecb):
        occ_b |= 1 << p
    _ = cistring.str2addr(norb, neleca, occ_a)
    _ = cistring.str2addr(norb, nelecb, occ_b)
    return SparseCIVector({(int(occ_a), int(occ_b)): 1.0})



def full_hdiag_map(*, hdiag_full: np.ndarray, norb: int, nelec: tuple[int, int]) -> dict[DetKey, float]:
    from pyscf.fci import cistring

    neleca, nelecb = nelec
    out: dict[DetKey, float] = {}
    for ia in range(hdiag_full.shape[0]):
        stra = int(cistring.addr2str(norb, neleca, int(ia)))
        for ib in range(hdiag_full.shape[1]):
            strb = int(cistring.addr2str(norb, nelecb, int(ib)))
            out[(stra, strb)] = float(hdiag_full[int(ia), int(ib)])
    return out
