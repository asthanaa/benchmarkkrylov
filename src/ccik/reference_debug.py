from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .core import _ritz_energy
from .params import CCIKParams
from .propagation import (
    KrylovStepArtifacts,
    KrylovStepDiagnostics,
    analyze_krylov_step,
    krylov_step_diagnostics_as_dict,
)
from .selection import make_full_hdiag_map
from .sparse_ci import (
    SparseCIVector,
    add_scaled_sparse,
    make_hf_sparse_vector,
    sparse_norm,
    union_support,
)

DetKey = tuple[int, int]


@dataclass(frozen=True)
class OracleStepComparison:
    step: int
    truncated: dict[str, object]
    exact: dict[str, object]
    qnext_l2_diff: float | None
    qnext_support_overlap: int | None
    qnext_support_union: int | None
    projected_support_overlap: int | None
    projected_support_union: int | None


def _sorted_sparse_items(vec: SparseCIVector, limit: int | None = None) -> list[dict[str, object]]:
    ranked = sorted(vec.coeffs.items(), key=lambda item: abs(item[1]), reverse=True)
    if limit is not None:
        ranked = ranked[: int(limit)]
    return [
        {"stra": int(key[0]), "strb": int(key[1]), "coef": float(val), "abs_coef": float(abs(val))}
        for key, val in ranked
    ]


def _sorted_det_keys(keys: set[DetKey], limit: int | None = None) -> list[dict[str, int]]:
    ranked = sorted(keys)
    if limit is not None:
        ranked = ranked[: int(limit)]
    return [{"stra": int(stra), "strb": int(strb)} for stra, strb in ranked]


def _artifacts_to_dict(
    diag: KrylovStepDiagnostics,
    artifacts: KrylovStepArtifacts,
    ws,
    q_next: SparseCIVector,
    *,
    topn: int,
) -> dict[str, object]:
    out = krylov_step_diagnostics_as_dict(diag)
    out.update(
        {
            "requested_selector": artifacts.requested_selector,
            "used_selector": artifacts.used_selector,
            "candidate_pool_size": len(artifacts.candidate_pool),
            "candidate_support_size": len(ws.candidate_support),
            "parent_support_size": len(artifacts.parent_support),
            "score_support_size": len(artifacts.score_support),
            "tmp_keep": int(artifacts.tmp_keep),
            "preserve_cap": int(artifacts.preserve_cap),
            "candidate_pool_top": _sorted_det_keys(artifacts.candidate_pool, limit=topn),
            "parent_support_top": _sorted_det_keys(artifacts.parent_support, limit=topn),
            "score_support_top": _sorted_det_keys(artifacts.score_support, limit=topn),
            "projected_top": _sorted_sparse_items(ws.v_proj, limit=topn),
            "score_vector_top": _sorted_sparse_items(artifacts.score_vector, limit=topn),
            "v_temp_top": _sorted_sparse_items(artifacts.v_temp, limit=topn),
            "orthogonalized_residual_top": _sorted_sparse_items(artifacts.orthogonalized_residual, limit=topn),
            "compressed_top": _sorted_sparse_items(artifacts.compressed, limit=topn),
            "q_next_top": _sorted_sparse_items(q_next, limit=topn),
        }
    )
    return out


def compare_single_krylov_step(
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
    step: int = 0,
    topn: int = 12,
) -> OracleStepComparison:
    if hdiag_full is None:
        hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)

    q_next_trunc, diag_trunc, ws_trunc, art_trunc = analyze_krylov_step(
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

    if not getattr(params, "exact_hq_diagnostics", False):
        trunc_support = q_next_trunc.support()
        projected_trunc = ws_trunc.v_proj.support()
        return OracleStepComparison(
            step=int(step),
            truncated=_artifacts_to_dict(diag_trunc, art_trunc, ws_trunc, q_next_trunc, topn=topn),
            exact={},
            qnext_l2_diff=None,
            qnext_support_overlap=len(trunc_support),
            qnext_support_union=len(trunc_support),
            projected_support_overlap=len(projected_trunc),
            projected_support_union=len(projected_trunc),
        )

    exact_params = replace(params, candidate_mode="full", selector_mode=getattr(params, "selector_mode", "abs_residual"), include_current_support=True, exact_hq_diagnostics=True)
    q_next_exact, diag_exact, ws_exact, art_exact = analyze_krylov_step(
        q_k,
        basis,
        h1=h1,
        eri8=eri8,
        h2eff=h2eff,
        norb=norb,
        nelec=nelec,
        params=exact_params,
        hdiag_full=hdiag_full,
        rng=rng,
    )

    qnext_diff = sparse_norm(add_scaled_sparse(q_next_exact, -1.0, q_next_trunc))
    trunc_support = q_next_trunc.support()
    exact_support = q_next_exact.support()
    projected_trunc = ws_trunc.v_proj.support()
    projected_exact = ws_exact.v_proj.support()

    return OracleStepComparison(
        step=int(step),
        truncated=_artifacts_to_dict(diag_trunc, art_trunc, ws_trunc, q_next_trunc, topn=topn),
        exact=_artifacts_to_dict(diag_exact, art_exact, ws_exact, q_next_exact, topn=topn),
        qnext_l2_diff=float(qnext_diff),
        qnext_support_overlap=len(trunc_support & exact_support),
        qnext_support_union=len(trunc_support | exact_support),
        projected_support_overlap=len(projected_trunc & projected_exact),
        projected_support_union=len(projected_trunc | projected_exact),
    )


def trace_ccik_run(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKParams | None = None,
    max_steps: int | None = None,
    topn: int = 12,
    output_path: str | Path | None = None,
) -> list[dict[str, object]]:
    if params is None:
        params = CCIKParams(exact_hq_diagnostics=True)

    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)
    q0 = make_hf_sparse_vector(norb=norb, nelec=nelec)
    Q: list[SparseCIVector] = [q0]
    traces: list[dict[str, object]] = []
    nsteps = int(params.m - 1 if max_steps is None else min(max_steps, params.m - 1))

    for step in range(nsteps):
        comp = compare_single_krylov_step(
            Q[-1],
            Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
            step=step,
            topn=topn,
        )
        row = asdict(comp)
        traces.append(row)
        q_next, _, _, _ = analyze_krylov_step(
            Q[-1],
            Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        if not q_next.coeffs:
            break
        Q.append(q_next)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for row in traces:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    return traces


def summarize_trace(traces: list[dict[str, object]]) -> dict[str, object]:
    if not traces:
        return {"n_steps": 0}
    qdiff = [float(row["qnext_l2_diff"]) for row in traces if row.get("qnext_l2_diff") is not None]
    out = {
        "n_steps": len(traces),
    }
    if qdiff:
        out["max_qnext_l2_diff"] = max(qdiff)
        out["mean_qnext_l2_diff"] = float(np.mean(qdiff))
    else:
        out["max_qnext_l2_diff"] = None
        out["mean_qnext_l2_diff"] = None
    return out


def trace_ccik_run_with_energy(
    h1: np.ndarray,
    eri8: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    *,
    params: CCIKParams | None = None,
    max_steps: int | None = None,
    topn: int = 12,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    traces = trace_ccik_run(h1, eri8, norb, nelec, params=params, max_steps=max_steps, topn=topn, output_path=output_path)
    if params is None:
        params = CCIKParams(exact_hq_diagnostics=True)
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(h1, eri8, norb, nelec, fac=0.5)  # type: ignore[arg-type]
    q0 = make_hf_sparse_vector(norb=norb, nelec=nelec)
    Q = [q0]
    hdiag_full = make_full_hdiag_map(h1=h1, eri8=eri8, norb=norb, nelec=nelec)
    for _ in traces:
        q_next, _, _, _ = analyze_krylov_step(
            Q[-1],
            Q,
            h1=h1,
            eri8=eri8,
            h2eff=h2eff,
            norb=norb,
            nelec=nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        if not q_next.coeffs:
            break
        Q.append(q_next)
    e0, _, _ = _ritz_energy(Q=Q, h1=h1, eri8=eri8, h2eff=h2eff, norb=norb, nelec=nelec, params=params, hdiag_full=hdiag_full)
    summary = summarize_trace(traces)
    summary["ritz_energy"] = float(e0)
    summary["basis_size"] = len(Q)
    summary["ndet_union"] = len(union_support(Q))
    return {"traces": traces, "summary": summary}
