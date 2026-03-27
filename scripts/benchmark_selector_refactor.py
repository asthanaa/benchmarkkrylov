"""Benchmark selector modes for the matvec-first CCIK refactor.

This script compares the legacy PT2-style selector against the new matvec-first
selectors on exact-reference PySCF CAS problems, focusing on N2 CAS(10,10) and
optionally larger / additional cases.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_stochastic,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKParams, CCIKStochasticParams, CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from n2_utils import make_n2_atom  # noqa: E402


@dataclass(frozen=True)
class CaseSpec:
    name: str
    family: str
    basis: str
    cas: CASSpec
    Rs: tuple[float, ...]
    charge: int = 0
    spin: int = 0


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


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="n2_cas10_10",
        family="n2",
        basis="6-31g",
        cas=CASSpec(ncas=10, nelecas=10, ncore=2),
        Rs=(1.10, 1.80, 2.85),
    ),
    CaseSpec(
        name="n2_cas12_12",
        family="n2",
        basis="6-31g",
        cas=CASSpec(ncas=12, nelecas=12, ncore=1),
        Rs=(1.10, 2.50),
    ),
)

METHODS: tuple[str, ...] = ("ccik", "ccik_thick")
SELECTORS: tuple[str, ...] = ("pt2_legacy", "abs_hq", "abs_residual")


def _build_case_point(case: CaseSpec, R: float) -> PointData:
    if case.family != "n2":
        raise ValueError(f"Unsupported family {case.family!r}")
    mol = make_mol_pyscf(
        atom=make_n2_atom(float(R)),
        basis=case.basis,
        unit="Angstrom",
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


def _base_params(method: str, selector_mode: str) -> Any:
    common = dict(
        selector_mode=selector_mode,
        selector_depth=1,
        generator_keep=256,
        generator_abs_coeff_cutoff=0.0,
        max_parents_for_expansion=256,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=2048,
        exact_hq_diagnostics=True,
        orth_tol=1e-12,
        verbose=False,
    )
    if method == "ccik":
        return CCIKParams(
            m=32,
            nadd=2000,
            nkeep=12000,
            Kv=4000,
            candidate_mode="selected_ci",
            selector_ci_coeff_cutoff=5e-4,
            selector_cutoff=5e-4,
            **common,
        )
    if method == "ccik_thick":
        return CCIKThickRestartParams(
            m_cycle=5,
            ncycles=10,
            nroot=2,
            tol=1e-6,
            nadd=4000,
            nkeep=8000,
            Kv=4000,
            candidate_mode="selected_ci",
            selector_ci_coeff_cutoff=5e-4,
            selector_cutoff=5e-4,
            **common,
        )
    if method == "ccik_stochastic":
        return CCIKStochasticParams(
            m=24,
            nadd=2000,
            nkeep=12000,
            Kv=2000,
            candidate_mode="stochastic",
            selector_depth=2,
            n_walkers=1500,
            seed=0,
            parent_power=1.0,
            p_double=0.6,
            mixed_double_weight=1.0,
            eps_denom=1e-12,
            **common,
        )
    raise ValueError(f"Unknown method {method!r}")


def _run_method(method: str, point: PointData, params: Any) -> dict[str, object]:
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    if method == "ccik":
        e_alg_cas = ccik_ground_energy_dense(point.h1, point.eri8, point.ncas, point.nelec, params=params, stats=stats)
    elif method == "ccik_thick":
        e_alg_cas = ccik_ground_energy_dense_thick_restart(
            point.h1,
            point.eri8,
            point.ncas,
            point.nelec,
            params=params,
            stats=stats,
        )
    elif method == "ccik_stochastic":
        e_alg_cas = ccik_ground_energy_stochastic(
            point.h1,
            point.eri8,
            point.ncas,
            point.nelec,
            params=params,
            stats=stats,
        )
    else:
        raise ValueError(f"Unknown method {method!r}")

    runtime_s = float(time.perf_counter() - t0)
    e_alg_total = float(point.ecore + e_alg_cas)
    abs_err = abs(e_alg_total - point.e_fci_total)
    last_diag = stats.get("last_step_diagnostics", {})
    if not isinstance(last_diag, dict):
        last_diag = {}

    return {
        "runtime_s": runtime_s,
        "e_alg_total": e_alg_total,
        "abs_err": abs_err,
        "signed_err": e_alg_total - point.e_fci_total,
        "ndet_sum": int(stats.get("ndet_sum", -1)),
        "ndet_union": int(stats.get("ndet_union", -1)),
        "m_eff": int(stats.get("m_eff", -1)),
        "last_selector_mode_used": last_diag.get("selector_mode_used"),
        "last_candidate_mode_used": last_diag.get("candidate_mode_used"),
        "last_relative_matvec_error": last_diag.get("relative_matvec_error"),
        "last_discarded_weight": last_diag.get("discarded_weight"),
        "last_orth_residual_norm": last_diag.get("orthogonalized_residual_norm"),
        "last_post_compression_residual_norm": last_diag.get("post_compression_residual_norm"),
        "last_num_candidates_generated": last_diag.get("num_candidates_generated"),
        "last_num_tmp_kept": last_diag.get("num_tmp_kept"),
        "last_num_final_kept": last_diag.get("num_final_kept"),
    }


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


def _selector_summary(rows: list[dict[str, object]], selector: str) -> tuple[float, float, float]:
    subset = [row for row in rows if row["selector_mode"] == selector]
    errs = [float(row["abs_err"]) for row in subset]
    times = [float(row["runtime_s"]) for row in subset]
    return max(errs), float(np.mean(errs)), float(np.sum(times))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark selector modes on N2 CAS cases")
    parser.add_argument(
        "--outdir",
        default=str(_ROOT / "experiments" / "selector_refactor_benchmark_20260320"),
        help="Directory for CSV summaries and notes",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[case.name for case in CASES],
        help="Subset of benchmark cases to run",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHODS),
        help="Subset of methods to run",
    )
    parser.add_argument(
        "--selectors",
        nargs="*",
        default=list(SELECTORS),
        help="Subset of selector modes to run",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_cases = [case for case in CASES if case.name in set(args.cases)]
    selected_methods = [method for method in (*METHODS, "ccik_stochastic") if method in set(args.methods)]
    selected_selectors = [selector for selector in SELECTORS if selector in set(args.selectors)]

    rows: list[dict[str, object]] = []
    summary_lines: list[str] = ["# Selector Refactor Benchmark", ""]

    for case in selected_cases:
        summary_lines.append(f"## {case.name}")
        summary_lines.append("")
        points: dict[float, PointData] = {}
        for R in case.Rs:
            t0 = time.perf_counter()
            print(f"Preparing {case.name} at R={R:.2f} Å", flush=True)
            points[R] = _build_case_point(case, R)
            print(
                f"  built in {time.perf_counter() - t0:.2f}s | CAS-FCI={points[R].e_fci_total:+.10f}",
                flush=True,
            )

        for method in selected_methods:
            summary_lines.append(f"### {method}")
            summary_lines.append("")
            for selector in selected_selectors:
                print(f"\n[{case.name}] [{method}] [{selector}]", flush=True)
                params = _base_params(method, selector)
                for R in case.Rs:
                    point = points[R]
                    metrics = _run_method(method, point, params)
                    row = {
                        "case": case.name,
                        "method": method,
                        "selector_mode": selector,
                        "R": R,
                        "e_fci_total": point.e_fci_total,
                        **asdict(params),
                        **metrics,
                    }
                    rows.append(row)
                    print(
                        f"  R={R:.2f}: E={float(metrics['e_alg_total']):+.10f} "
                        f"|Δ|={float(metrics['abs_err']):.3e} t={float(metrics['runtime_s']):.2f}s "
                        f"ndet={int(metrics['ndet_sum'])} matvec_err={metrics['last_relative_matvec_error']}",
                        flush=True,
                    )

                max_err, mean_err, total_t = _selector_summary(
                    [row for row in rows if row["case"] == case.name and row["method"] == method],
                    selector,
                )
                summary_lines.append(
                    f"- `{selector}`: max |ΔE| = `{max_err:.3e} Eh`, mean |ΔE| = `{mean_err:.3e} Eh`, total runtime = `{total_t:.2f} s`"
                )
            summary_lines.append("")

    _write_csv(outdir / "results.csv", rows)
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nWrote: {outdir / 'results.csv'}")
    print(f"Wrote: {outdir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
