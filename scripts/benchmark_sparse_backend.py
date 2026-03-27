"""Benchmark and tune the sparse CCIK backends on representative N2 CAS cases."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    basis: str
    cas: CASSpec
    tune_R: float
    validate_Rs: tuple[float, ...]
    target_err: float = 1e-3


@dataclass(frozen=True)
class PointData:
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
        basis="6-31g",
        cas=CASSpec(ncas=10, nelecas=10, ncore=2),
        tune_R=2.85,
        validate_Rs=(1.10, 1.80, 2.85),
        target_err=1e-3,
    ),
    CaseSpec(
        name="n2_cas12_12",
        basis="6-31g",
        cas=CASSpec(ncas=12, nelecas=12, ncore=1),
        tune_R=2.50,
        validate_Rs=(1.10, 2.50),
        target_err=2e-3,
    ),
)

METHODS: tuple[str, ...] = ("ccik", "ccik_thick", "ccik_stochastic")


def _build_case_point(case: CaseSpec, R: float) -> PointData:
    mol = make_mol_pyscf(
        atom=make_n2_atom(float(R)),
        basis=case.basis,
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=case.cas)
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    return PointData(R=float(R), h1=h1, eri8=eri8, ecore=float(ecore), ncas=int(ncas), nelec=nelec, e_fci_total=e_fci_total)


def _base_params(method: str) -> Any:
    if method == "ccik":
        return CCIKParams(
            m=32,
            nadd=2000,
            nkeep=8000,
            Kv=2000,
            selector_mode="cipsi",
            selector_depth=1,
            orth_tol=1e-12,
            verbose=False,
        )
    if method == "ccik_thick":
        return CCIKThickRestartParams(
            m_cycle=5,
            ncycles=10,
            nroot=2,
            tol=1e-6,
            nadd=2000,
            nkeep=8000,
            Kv=2000,
            selector_mode="cipsi",
            selector_depth=1,
            orth_tol=1e-12,
            verbose=False,
        )
    if method == "ccik_stochastic":
        return CCIKStochasticParams(
            m=24,
            nadd=4000,
            nkeep=12000,
            Kv=2000,
            selector_mode="stochastic",
            selector_depth=2,
            n_walkers=1500,
            seed=0,
            parent_power=1.0,
            p_double=0.6,
            mixed_double_weight=1.0,
            eps_denom=1e-12,
            orth_tol=1e-12,
            verbose=False,
        )
    raise ValueError(f"Unknown method {method!r}")


def _search_plan(method: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if method in ("ccik", "ccik_thick"):
        return (
            ("selector_depth", (1, 2)),
            ("nadd", (1000, 2000, 4000)),
            ("nkeep", (6000, 8000, 12000)),
            ("Kv", (1000, 2000, 4000)),
        )
    return (
        ("selector_depth", (1, 2)),
        ("nadd", (2000, 4000)),
        ("nkeep", (8000, 12000)),
        ("Kv", (1000, 2000)),
    )


def _run_method(method: str, point: PointData, params: Any) -> dict[str, float | int]:
    stats: dict[str, int] = {}
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
    out: dict[str, float | int] = {
        "runtime_s": runtime_s,
        "e_alg_total": e_alg_total,
        "abs_err": abs_err,
        "ndet_sum": int(stats.get("ndet_sum", -1)),
        "ndet_union": int(stats.get("ndet_union", -1)),
        "m_eff": int(stats.get("m_eff", -1)),
    }
    return out


def _candidate_rank(metrics: dict[str, float | int], target_err: float) -> tuple[float, float, float, float]:
    abs_err = float(metrics["abs_err"])
    runtime_s = float(metrics["runtime_s"])
    ndet_sum = float(metrics["ndet_sum"])
    return (0.0 if abs_err <= target_err else 1.0, abs_err, runtime_s, ndet_sum)


def _validate_rank(rows: list[dict[str, float | int]], target_err: float) -> tuple[float, float, float, float]:
    errs = [float(row["abs_err"]) for row in rows]
    times = [float(row["runtime_s"]) for row in rows]
    return (
        0.0 if max(errs) <= target_err else 1.0,
        max(errs),
        float(np.mean(errs)),
        float(np.sum(times)),
    )


def _format_params(params: Any) -> str:
    parts = []
    for key, value in asdict(params).items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _tune_case_method(
    case: CaseSpec,
    method: str,
    points: dict[float, PointData],
    *,
    trial_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> Any:
    current = _base_params(method)
    tune_point = points[case.tune_R]

    print(f"\n[{case.name}] [{method}] base: {_format_params(current)}", flush=True)
    base_metrics = _run_method(method, tune_point, current)
    trial_rows.append(
        {
            "case": case.name,
            "method": method,
            "stage": "base",
            "knob": "base",
            "value": "base",
            "R": tune_point.R,
            **asdict(current),
            **base_metrics,
            "target_err": case.target_err,
        }
    )
    print(
        f"  base @R={tune_point.R:.2f}: err={float(base_metrics['abs_err']):.3e} t={float(base_metrics['runtime_s']):.2f}s",
        flush=True,
    )

    for knob, values in _search_plan(method):
        best_params = current
        best_metrics = _run_method(method, tune_point, current)
        best_rank = _candidate_rank(best_metrics, case.target_err)
        for value in values:
            candidate = replace(current, **{knob: value})
            metrics = _run_method(method, tune_point, candidate)
            trial_rows.append(
                {
                    "case": case.name,
                    "method": method,
                    "stage": f"tune_{knob}",
                    "knob": knob,
                    "value": value,
                    "R": tune_point.R,
                    **asdict(candidate),
                    **metrics,
                    "target_err": case.target_err,
                }
            )
            rank = _candidate_rank(metrics, case.target_err)
            print(
                f"  {knob}={value}: err={float(metrics['abs_err']):.3e} t={float(metrics['runtime_s']):.2f}s",
                flush=True,
            )
            if rank < best_rank:
                best_rank = rank
                best_metrics = metrics
                best_params = candidate
        current = best_params
        print(
            f"  -> keep {knob}={getattr(current, knob)} | err={float(best_metrics['abs_err']):.3e} t={float(best_metrics['runtime_s']):.2f}s",
            flush=True,
        )

    rows_for_rank: list[dict[str, float | int]] = []
    for R in case.validate_Rs:
        point = points[R]
        metrics = _run_method(method, point, current)
        validation_rows.append(
            {
                "case": case.name,
                "method": method,
                "R": R,
                **asdict(current),
                **metrics,
                "e_fci_total": point.e_fci_total,
                "target_err": case.target_err,
            }
        )
        rows_for_rank.append(metrics)
        print(
            f"  validate R={R:.2f}: err={float(metrics['abs_err']):.3e} t={float(metrics['runtime_s']):.2f}s ndet={int(metrics['ndet_sum'])}",
            flush=True,
        )

    rank = _validate_rank(rows_for_rank, case.target_err)
    print(
        f"  summary: max_err={rank[1]:.3e} mean_err={rank[2]:.3e} total_t={rank[3]:.2f}s target={case.target_err:.1e}",
        flush=True,
    )
    return current


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune sparse CCIK backends on representative N2 cases")
    parser.add_argument(
        "--outdir",
        default=str(_ROOT / "experiments" / "sparse_backend_benchmark"),
        help="Directory for CSV summaries and notes",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[case.name for case in CASES],
        help="Subset of cases to run",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=list(METHODS),
        help="Subset of methods to run",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected_cases = [case for case in CASES if case.name in set(args.cases)]
    selected_methods = [method for method in METHODS if method in set(args.methods)]

    trial_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    summary_lines: list[str] = ["# Sparse Backend Benchmark", ""]

    for case in selected_cases:
        summary_lines.append(f"## {case.name}")
        summary_lines.append("")
        all_Rs = tuple(sorted(set((case.tune_R, *case.validate_Rs))))
        points: dict[float, PointData] = {}
        for R in all_Rs:
            t0 = time.perf_counter()
            print(f"\nPreparing {case.name} at R={R:.2f} Å", flush=True)
            points[R] = _build_case_point(case, R)
            print(
                f"  built in {time.perf_counter() - t0:.2f}s | CAS-FCI={points[R].e_fci_total:+.10f}",
                flush=True,
            )

        for method in selected_methods:
            tuned = _tune_case_method(case, method, points, trial_rows=trial_rows, validation_rows=validation_rows)
            rows = [
                row
                for row in validation_rows
                if row["case"] == case.name and row["method"] == method
            ]
            errs = [float(row["abs_err"]) for row in rows]
            times = [float(row["runtime_s"]) for row in rows]
            summary_lines.append(f"### {method}")
            summary_lines.append("")
            summary_lines.append(f"- tuned params: `{_format_params(tuned)}`")
            summary_lines.append(f"- max |ΔE| on validation set: `{max(errs):.3e} Eh`")
            summary_lines.append(f"- mean |ΔE| on validation set: `{float(np.mean(errs)):.3e} Eh`")
            summary_lines.append(f"- total validation runtime: `{float(np.sum(times)):.2f} s`")
            summary_lines.append("")

    _write_csv(outdir / "trial_results.csv", trial_rows)
    _write_csv(outdir / "validation_results.csv", validation_rows)
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nWrote: {outdir / 'trial_results.csv'}")
    print(f"Wrote: {outdir / 'validation_results.csv'}")
    print(f"Wrote: {outdir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
