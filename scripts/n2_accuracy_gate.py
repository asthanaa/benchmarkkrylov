"""Exact-FCI N2 CAS(10,10) accuracy gate for maintained CCIK solvers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKParams, CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402
from n2_utils import make_n2_atom  # noqa: E402


R_POINTS: tuple[float, ...] = (1.10, 1.80, 2.85)
CAS = CASSpec(ncas=10, nelecas=10, ncore=2)


@dataclass(frozen=True)
class PointData:
    R: float
    h1: Any
    eri8: Any
    ecore: float
    ncas: int
    nelec: tuple[int, int]
    e_fci_total: float


def _build_point(R: float) -> PointData:
    mol = make_mol_pyscf(
        atom=make_n2_atom(float(R)),
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=CAS)
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    return PointData(R=float(R), h1=h1, eri8=eri8, ecore=float(ecore), ncas=int(ncas), nelec=nelec, e_fci_total=e_fci_total)


def _dense_ladder() -> list[tuple[str, CCIKParams]]:
    common = dict(
        candidate_mode="selected_ci",
        selector_mode="abs_residual",
        selector_depth=1,
        generator_abs_coeff_cutoff=0.0,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=2048,
        exact_hq_diagnostics=False,
        selector_ci_coeff_cutoff=5e-4,
        selector_cutoff=5e-4,
        orth_tol=1e-12,
        verbose=False,
    )
    return [
        ("dense_r1", CCIKParams(m=32, nadd=2000, nkeep=12000, Kv=4000, generator_keep=256, **common)),
        ("dense_r2", CCIKParams(m=40, nadd=4000, nkeep=16000, Kv=6000, generator_keep=512, max_parents_for_expansion=512, **common)),
        ("dense_r3", CCIKParams(m=48, nadd=4000, nkeep=24000, Kv=8000, generator_keep=512, max_parents_for_expansion=512, **common)),
    ]


def _thick_ladder() -> list[tuple[str, CCIKThickRestartParams]]:
    common = dict(
        candidate_mode="selected_ci",
        selector_mode="abs_residual",
        selector_depth=1,
        generator_abs_coeff_cutoff=0.0,
        include_current_support=True,
        tmp_keep_factor=4,
        preserve_prev_support=True,
        preserve_prev_support_cap=2048,
        exact_hq_diagnostics=False,
        selector_ci_coeff_cutoff=5e-4,
        selector_cutoff=5e-4,
        orth_tol=1e-12,
        verbose=False,
    )
    return [
        (
            "thick_r1",
            CCIKThickRestartParams(
                m_cycle=5,
                ncycles=10,
                nroot=2,
                tol=1e-6,
                nadd=4000,
                nkeep=8000,
                Kv=4000,
                generator_keep=256,
                max_parents_for_expansion=256,
                **common,
            ),
        ),
        (
            "thick_r2",
            CCIKThickRestartParams(
                m_cycle=10,
                ncycles=10,
                nroot=2,
                tol=1e-6,
                nadd=4000,
                nkeep=12000,
                Kv=6000,
                generator_keep=256,
                max_parents_for_expansion=256,
                **common,
            ),
        ),
        (
            "thick_r3",
            CCIKThickRestartParams(
                m_cycle=10,
                ncycles=20,
                nroot=3,
                tol=1e-6,
                nadd=4000,
                nkeep=16000,
                Kv=6000,
                generator_keep=512,
                max_parents_for_expansion=512,
                **common,
            ),
        ),
        (
            "thick_r4",
            CCIKThickRestartParams(
                m_cycle=15,
                ncycles=20,
                nroot=3,
                tol=1e-6,
                nadd=8000,
                nkeep=24000,
                Kv=8000,
                generator_keep=512,
                max_parents_for_expansion=512,
                **common,
            ),
        ),
    ]


def _run_dense(point: PointData, params: CCIKParams) -> dict[str, object]:
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    e_cas = ccik_ground_energy_dense(point.h1, point.eri8, point.ncas, point.nelec, params=params, stats=stats)
    runtime_s = float(time.perf_counter() - t0)
    e_total = float(point.ecore + e_cas)
    return {
        "runtime_s": runtime_s,
        "e_total": e_total,
        "abs_err": abs(e_total - point.e_fci_total),
        "signed_err": e_total - point.e_fci_total,
        "ndet_sum": int(stats.get("ndet_sum", -1)),
        "ndet_union": int(stats.get("ndet_union", -1)),
        "m_eff": int(stats.get("m_eff", -1)),
    }


def _run_thick(point: PointData, params: CCIKThickRestartParams) -> dict[str, object]:
    stats: dict[str, object] = {}
    t0 = time.perf_counter()
    e_cas = ccik_ground_energy_dense_thick_restart(
        point.h1,
        point.eri8,
        point.ncas,
        point.nelec,
        params=params,
        stats=stats,
    )
    runtime_s = float(time.perf_counter() - t0)
    e_total = float(point.ecore + e_cas)
    return {
        "runtime_s": runtime_s,
        "e_total": e_total,
        "abs_err": abs(e_total - point.e_fci_total),
        "signed_err": e_total - point.e_fci_total,
        "ndet_sum": int(stats.get("ndet_sum", -1)),
        "ndet_union": int(stats.get("ndet_union", -1)),
        "m_eff": int(stats.get("m_eff", -1)),
    }


def _max_err(rows: list[dict[str, object]], label: str) -> float:
    return max(float(row["abs_err"]) for row in rows if row["rung"] == label)


def _energy_map(rows: list[dict[str, object]], label: str) -> dict[float, float]:
    return {float(row["R"]): float(row["e_total"]) for row in rows if row["rung"] == label}


def _is_stable(rows: list[dict[str, object]], rung_a: str, rung_b: str, tol: float) -> bool:
    map_a = _energy_map(rows, rung_a)
    map_b = _energy_map(rows, rung_b)
    return all(abs(map_a[R] - map_b[R]) < tol for R in R_POINTS)


def _choose_rung(rows: list[dict[str, object]], ladder_labels: list[str], *, gate_tol: float, stability_tol: float) -> str:
    for idx, label in enumerate(ladder_labels):
        if _max_err(rows, label) > gate_tol:
            continue
        if idx == len(ladder_labels) - 1:
            return label
        next_label = ladder_labels[idx + 1]
        if _is_stable(rows, label, next_label, stability_tol):
            return label
    return ladder_labels[-1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="ascii") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact-FCI N2 CAS(10,10) accuracy gate")
    p.add_argument(
        "--outdir",
        type=Path,
        default=_ROOT / "experiments" / "n2_accuracy_gate_20260320",
    )
    p.add_argument("--gate-tol-eh", type=float, default=1.0e-3)
    p.add_argument("--stability-tol-eh", type=float, default=1.0e-4)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    points = {R: _build_point(R) for R in R_POINTS}
    results: list[dict[str, object]] = []

    dense_ladder = _dense_ladder()
    for label, params in dense_ladder:
        for R in R_POINTS:
            metrics = _run_dense(points[R], params)
            results.append(
                {
                    "method": "ccik",
                    "rung": label,
                    "R": R,
                    "e_fci_total": points[R].e_fci_total,
                    **asdict(params),
                    **metrics,
                }
            )

    thick_ladder = _thick_ladder()
    for label, params in thick_ladder:
        for R in R_POINTS:
            metrics = _run_thick(points[R], params)
            results.append(
                {
                    "method": "ccik_thick",
                    "rung": label,
                    "R": R,
                    "e_fci_total": points[R].e_fci_total,
                    **asdict(params),
                    **metrics,
                }
            )

    dense_choice = _choose_rung(results, [label for label, _ in dense_ladder], gate_tol=args.gate_tol_eh, stability_tol=args.stability_tol_eh)
    thick_choice = _choose_rung(results, [label for label, _ in thick_ladder], gate_tol=args.gate_tol_eh, stability_tol=args.stability_tol_eh)

    chosen = {"ccik": dense_choice, "ccik_thick": thick_choice}
    chosen_params = {
        "ccik": next(asdict(params) for label, params in dense_ladder if label == dense_choice),
        "ccik_thick": next(asdict(params) for label, params in thick_ladder if label == thick_choice),
    }

    gate = {
        method: {
            "chosen_rung": chosen[method],
            "max_abs_err_eh": max(float(row["abs_err"]) for row in results if row["method"] == method and row["rung"] == chosen[method]),
            "passed": max(float(row["abs_err"]) for row in results if row["method"] == method and row["rung"] == chosen[method]) <= args.gate_tol_eh,
        }
        for method in ("ccik", "ccik_thick")
    }
    gate["overall_passed"] = bool(gate["ccik"]["passed"] and gate["ccik_thick"]["passed"])

    _write_csv(outdir / "results.csv", results)
    (outdir / "selected_params.json").write_text(json.dumps(chosen_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (outdir / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_lines = [
        "# N2 Accuracy Gate",
        "",
        f"- Points: {', '.join(f'{R:.2f}' for R in R_POINTS)}",
        f"- Gate tol (Eh): {args.gate_tol_eh}",
        f"- Stability tol (Eh): {args.stability_tol_eh}",
        f"- Overall passed: {gate['overall_passed']}",
        "",
    ]
    for method in ("ccik", "ccik_thick"):
        summary_lines.append(
            f"- `{method}`: chosen `{gate[method]['chosen_rung']}`, max |ΔE| = `{gate[method]['max_abs_err_eh']:.6e} Eh`, passed = `{gate[method]['passed']}`"
        )
    (outdir / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(json.dumps(gate, indent=2))
    return 0 if gate["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
