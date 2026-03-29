"""Compare fixed-budget selector modes on small exact-reference molecules.

This benchmark is intentionally small and exact so we can compare determinant
selection choices without mixing in reference uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import build_cas_hamiltonian_pyscf, exact_cas_fci_energy_pyscf, make_mol_pyscf  # noqa: E402
from ccik.params import CCIKParams, CCIKThickRestartParams  # noqa: E402
from ccik.propagation import build_next_krylov_vector  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from ccik.ritz import build_exact_projected_matrices, generalized_eigh, solve_projected_ritz  # noqa: E402
from ccik.selection import candidate_mode_from_params, make_full_hdiag_map  # noqa: E402
from ccik.sparse_ci import linear_combination_sparse, make_hf_sparse_vector, normalize_sparse  # noqa: E402
from ccik.thick_restart import _ccik_cycle_sparse_multi_start, _compress_with_stage1, gram_schmidt_list  # noqa: E402


@dataclass(frozen=True)
class CaseSpec:
    name: str
    atom_builder: Callable[[float], str]
    basis: str
    cas: CASSpec
    Rs: tuple[float, ...]
    fixed_store_keep: int
    fixed_propagate_keep: int
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"


@dataclass(frozen=True)
class PointData:
    case: str
    R: float
    h1: np.ndarray
    eri8: np.ndarray
    ecore: float
    ncas: int
    nelec: tuple[int, int]
    e_fci_total: float


def _h2_atom(r_ang: float) -> str:
    return f"H 0 0 0; H 0 0 {float(r_ang):.8f}"


def _h4_atom(r_ang: float) -> str:
    return "; ".join(f"H 0 0 {float(i * r_ang):.8f}" for i in range(4))


def _n2_atom(r_ang: float) -> str:
    half = float(r_ang) / 2.0
    return f"N 0 0 {-half:.8f}; N 0 0 {half:.8f}"


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="h2_cas2_2",
        atom_builder=_h2_atom,
        basis="sto-3g",
        cas=CASSpec(ncas=2, nelecas=2, ncore=0),
        Rs=(0.74,),
        fixed_store_keep=4,
        fixed_propagate_keep=2,
    ),
    CaseSpec(
        name="h4_cas4_4",
        atom_builder=_h4_atom,
        basis="sto-3g",
        cas=CASSpec(ncas=4, nelecas=4, ncore=0),
        Rs=(1.50, 2.00),
        fixed_store_keep=8,
        fixed_propagate_keep=4,
    ),
    CaseSpec(
        name="n2_cas6_6",
        atom_builder=_n2_atom,
        basis="6-31g",
        cas=CASSpec(ncas=6, nelecas=6, ncore=2),
        Rs=(1.10, 2.00),
        fixed_store_keep=24,
        fixed_propagate_keep=8,
    ),
)

METHODS: tuple[str, ...] = ("ccik", "ccik_thick")
SELECTORS: tuple[str, ...] = ("abs_hq", "abs_residual", "pt2_legacy", "ground_state_pt2")
SCOPES: tuple[str, ...] = ("isolated", "realistic")


def _build_case_point(case: CaseSpec, R: float) -> PointData:
    mol = make_mol_pyscf(
        atom=case.atom_builder(float(R)),
        basis=case.basis,
        unit=case.unit,
        charge=case.charge,
        spin=case.spin,
        verbose=0,
    )
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=case.cas)
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    return PointData(
        case=case.name,
        R=float(R),
        h1=h1,
        eri8=eri8,
        ecore=float(ecore),
        ncas=int(ncas),
        nelec=nelec,
        e_fci_total=e_fci_total,
    )


def _common_kwargs(case: CaseSpec, selector_mode: str, scope: str) -> dict[str, object]:
    candidate_mode = "full" if scope == "isolated" else "selected_ci"
    return dict(
        compression_policy="fixed_budget",
        fixed_store_keep=case.fixed_store_keep,
        fixed_propagate_keep=case.fixed_propagate_keep,
        candidate_mode=candidate_mode,
        selector_mode=selector_mode,
        selector_depth=1,
        generator_keep=case.fixed_propagate_keep,
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=case.fixed_propagate_keep,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=False,
        selector_ci_coeff_cutoff=1e-6,
        selector_cutoff=1e-6,
        exact_hq_diagnostics=False,
        orth_tol=1e-12,
        verbose=False,
    )


def _dense_params(case: CaseSpec, selector_mode: str, scope: str) -> CCIKParams:
    return CCIKParams(
        m=12,
        nadd=case.fixed_store_keep,
        nkeep=case.fixed_store_keep,
        Kv=case.fixed_propagate_keep,
        **_common_kwargs(case, selector_mode, scope),
    )


def _thick_params(case: CaseSpec, selector_mode: str, scope: str) -> CCIKThickRestartParams:
    return CCIKThickRestartParams(
        m_cycle=9,
        ncycles=2,
        nroot=3,
        tol=1e-10,
        nadd=case.fixed_store_keep,
        nkeep=case.fixed_store_keep,
        Kv=case.fixed_propagate_keep,
        **_common_kwargs(case, selector_mode, scope),
    )


def _basis_stats(Q: list[Any]) -> tuple[int, int, int]:
    union: set[tuple[int, int]] = set()
    ndet_sum = 0
    for q in Q:
        supp = q.support()
        union.update(supp)
        ndet_sum += len(supp)
    return len(Q), ndet_sum, len(union)


def _final_metrics(
    Q: list[Any],
    point: PointData,
    params: Any,
    *,
    h2eff: np.ndarray,
    hdiag_full: dict[tuple[int, int], float] | None,
) -> dict[str, float]:
    approx = solve_projected_ritz(
        Q=Q,
        h1=point.h1,
        eri8=point.eri8,
        h2eff=h2eff,
        norb=point.ncas,
        nelec=point.nelec,
        params=params,
        hdiag_full=hdiag_full,
    )
    exact_Hproj, exact_Sproj, _ = build_exact_projected_matrices(Q, h2eff=h2eff, norb=point.ncas, nelec=point.nelec)
    exact_evals, _ = generalized_eigh(exact_Hproj, exact_Sproj)

    approx_total = float(point.ecore + approx.evals[0])
    exact_same_basis_total = float(point.ecore + exact_evals[0])
    hdiff = approx.Hproj - exact_Hproj
    return {
        "e_alg_total": approx_total,
        "abs_err": abs(approx_total - point.e_fci_total),
        "signed_err": approx_total - point.e_fci_total,
        "projected_h_fro_error": float(np.linalg.norm(hdiff, ord="fro")),
        "projected_h_maxabs_error": float(np.max(np.abs(hdiff))) if hdiff.size else 0.0,
        "same_basis_ritz_error": abs(approx_total - exact_same_basis_total),
        "same_basis_exact_total": exact_same_basis_total,
        "approx_ritz_total": approx_total,
    }


def _run_dense_method(point: PointData, params: CCIKParams) -> dict[str, object]:
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(point.h1, point.eri8, point.ncas, point.nelec, fac=0.5)  # type: ignore[arg-type]
    mode = candidate_mode_from_params(params, default="selected_ci")
    hdiag_full = make_full_hdiag_map(h1=point.h1, eri8=point.eri8, norb=point.ncas, nelec=point.nelec) if mode == "full" else None

    q0 = make_hf_sparse_vector(norb=point.ncas, nelec=point.nelec)
    Q = [q0]
    t0 = time.perf_counter()
    for k in range(params.m - 1):
        q_next, _, _ = build_next_krylov_vector(
            Q[k],
            Q,
            h1=point.h1,
            eri8=point.eri8,
            h2eff=h2eff,
            norb=point.ncas,
            nelec=point.nelec,
            params=params,
            hdiag_full=hdiag_full,
        )
        if not q_next.coeffs:
            break
        Q.append(q_next)
    runtime_s = float(time.perf_counter() - t0)
    m_eff, ndet_sum, ndet_union = _basis_stats(Q)
    return {
        "runtime_s": runtime_s,
        "m_eff": m_eff,
        "ndet_sum": ndet_sum,
        "ndet_union": ndet_union,
        **_final_metrics(Q, point, params, h2eff=h2eff, hdiag_full=hdiag_full),
    }


def _run_thick_method(point: PointData, params: CCIKThickRestartParams) -> dict[str, object]:
    from pyscf.fci import direct_spin1

    h2eff = direct_spin1.absorb_h1e(point.h1, point.eri8, point.ncas, point.nelec, fac=0.5)  # type: ignore[arg-type]
    mode = candidate_mode_from_params(params, default="selected_ci")
    hdiag_full = make_full_hdiag_map(h1=point.h1, eri8=point.eri8, norb=point.ncas, nelec=point.nelec) if mode == "full" else None

    starts = [make_hf_sparse_vector(norb=point.ncas, nelec=point.nelec)]
    Q: list[Any] = starts
    E_prev: float | None = None
    t0 = time.perf_counter()
    for _cyc in range(params.ncycles):
        evals, X, Q = _ccik_cycle_sparse_multi_start(
            point.h1,
            point.eri8,
            point.ncas,
            point.nelec,
            start_vecs=starts,
            params=params,
            hdiag_full=hdiag_full,
        )
        nroot_eff = min(params.nroot, X.shape[1])
        new_starts = []
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
        if E_prev is not None and abs(E0 - E_prev) < float(params.tol):
            break
        E_prev = E0
    runtime_s = float(time.perf_counter() - t0)
    m_eff, ndet_sum, ndet_union = _basis_stats(Q)
    return {
        "runtime_s": runtime_s,
        "m_eff": m_eff,
        "ndet_sum": ndet_sum,
        "ndet_union": ndet_union,
        **_final_metrics(Q, point, params, h2eff=h2eff, hdiag_full=hdiag_full),
    }


def _run_method(method: str, point: PointData, params: Any) -> dict[str, object]:
    if method == "ccik":
        return _run_dense_method(point, params)
    if method == "ccik_thick":
        return _run_thick_method(point, params)
    raise ValueError(f"Unknown method {method!r}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selector_summary(rows: list[dict[str, object]]) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    selectors = sorted({str(row["selector_mode"]) for row in rows})
    for selector in selectors:
        subset = [row for row in rows if row["selector_mode"] == selector]
        mean_abs_err = float(np.mean([float(row["abs_err"]) for row in subset]))
        mean_frob = float(np.mean([float(row["projected_h_fro_error"]) for row in subset]))
        summary[selector] = (mean_abs_err, mean_frob)
    return summary


def _winner(summary: dict[str, tuple[float, float]]) -> str:
    return min(summary, key=lambda mode: (summary[mode][0], summary[mode][1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fixed-budget selector modes on exact-reference small molecules")
    parser.add_argument(
        "--outdir",
        default=str(_ROOT / "experiments" / "fixed_budget_selector_benchmark_20260327"),
        help="Directory for CSV results and summary notes",
    )
    parser.add_argument("--cases", nargs="*", default=[case.name for case in CASES], help="Subset of cases to run")
    parser.add_argument("--methods", nargs="*", default=list(METHODS), help="Subset of methods to run")
    parser.add_argument("--selectors", nargs="*", default=list(SELECTORS), help="Subset of selectors to run")
    parser.add_argument("--scopes", nargs="*", default=list(SCOPES), help="Subset of scopes to run")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_cases = [case for case in CASES if case.name in set(args.cases)]
    selected_methods = [method for method in METHODS if method in set(args.methods)]
    selected_selectors = [selector for selector in SELECTORS if selector in set(args.selectors)]
    selected_scopes = [scope for scope in SCOPES if scope in set(args.scopes)]

    rows: list[dict[str, object]] = []
    summary_lines: list[str] = ["# Fixed-Budget Selector Benchmark", ""]

    for case in selected_cases:
        summary_lines.append(f"## {case.name}")
        summary_lines.append("")
        points: dict[float, PointData] = {}
        for R in case.Rs:
            t0 = time.perf_counter()
            print(f"Preparing {case.name} at R={R:.2f} Å", flush=True)
            points[R] = _build_case_point(case, R)
            print(
                f"  built in {time.perf_counter() - t0:.2f}s | exact CAS-FCI={points[R].e_fci_total:+.10f}",
                flush=True,
            )

        for scope in selected_scopes:
            summary_lines.append(f"### {scope}")
            summary_lines.append("")
            for method in selected_methods:
                summary_lines.append(f"#### {method}")
                summary_lines.append("")
                for selector in selected_selectors:
                    params = _dense_params(case, selector, scope) if method == "ccik" else _thick_params(case, selector, scope)
                    print(f"\n[{case.name}] [{scope}] [{method}] [{selector}]", flush=True)
                    for R in case.Rs:
                        point = points[R]
                        metrics = _run_method(method, point, params)
                        row = {
                            "case": case.name,
                            "scope": scope,
                            "method": method,
                            "selector_mode": selector,
                            "candidate_mode": candidate_mode_from_params(params, default="selected_ci"),
                            "R": R,
                            "e_fci_total": point.e_fci_total,
                            **asdict(params),
                            **metrics,
                        }
                        rows.append(row)
                        print(
                            f"  R={R:.2f}: E={float(metrics['e_alg_total']):+.10f} "
                            f"|Δ|={float(metrics['abs_err']):.3e} "
                            f"Hfro={float(metrics['projected_h_fro_error']):.3e} "
                            f"same-basis={float(metrics['same_basis_ritz_error']):.3e} "
                            f"t={float(metrics['runtime_s']):.2f}s",
                            flush=True,
                        )

                scope_rows = [
                    row
                    for row in rows
                    if row["case"] == case.name and row["scope"] == scope and row["method"] == method
                ]
                summary = _selector_summary(scope_rows)
                winner = _winner(summary)
                for selector, (mean_abs_err, mean_frob) in summary.items():
                    summary_lines.append(
                        f"- `{selector}`: mean |ΔE| = `{mean_abs_err:.3e} Eh`, mean H Frobenius error = `{mean_frob:.3e}`"
                    )
                summary_lines.append(f"- winner: `{winner}`")
                summary_lines.append("")

    realistic_winners: dict[str, str] = {}
    for method in selected_methods:
        realistic_rows = [row for row in rows if row["scope"] == "realistic" and row["method"] == method]
        if realistic_rows:
            realistic_winners[method] = _winner(_selector_summary(realistic_rows))

    summary_lines.append("## Recommendation")
    summary_lines.append("")
    if realistic_winners and all(winner == "ground_state_pt2" for winner in realistic_winners.values()):
        summary_lines.append("- `ground_state_pt2` wins the realistic fixed-budget benchmark for every deterministic solver tested.")
        summary_lines.append("- Recommendation: make `ground_state_pt2` the fixed-budget selector default.")
    else:
        summary_lines.append("- The realistic benchmark does not show a universal `ground_state_pt2` win across deterministic solvers.")
        summary_lines.append("- Recommendation: keep the current default selector and expose `ground_state_pt2` as an opt-in fixed-budget mode.")
    summary_lines.append("")

    _write_csv(outdir / "results.csv", rows)
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nWrote: {outdir / 'results.csv'}")
    print(f"Wrote: {outdir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
