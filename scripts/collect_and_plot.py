#!/usr/bin/env python
"""Collect QRTE/QKUD summaries and plot iterations versus Renyi entropy."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_PLOT_DIR = BENCHMARK_ROOT / "plots"
RUN_DIRS = {
    "qrte": BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.1",
    "qkud": BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.1",
}
SWEEP_DIRS = {
    "qrte": BENCHMARK_ROOT / "runs" / "qrte_natorb_delta_sweep",
    "qkud": BENCHMARK_ROOT / "runs" / "qkud_natorb_epsilon_sweep",
}
SWEEP_PARAMETER_NAMES = {
    "qrte": "delta_t",
    "qkud": "epsilon",
}
CANONICAL_SWEEP_VALUES = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
)
CANONICAL_SWEEP_VALUE_SET = {round(value, 10) for value in CANONICAL_SWEEP_VALUES}
OUTPUT_PREFIXES = {
    "qrte": "adapt_vs_qrte_iterations_vs_entropy",
    "qkud": "adapt_vs_qkud_iterations_vs_entropy",
}
ADAPT_OVERRIDE_SUMMARIES = {
    "H10_near_equi_sto3g_0": BENCHMARK_ROOT / "runs" / "adapt_h10_qcant" / "summary.json",
}
ADAPT_FIXED_OVERRIDES = {
    "H10_near_equi_sto3g_0": {
        "iteration": 324.5,
        "source": "paper_fig1_extrapolated",
        "note": "H10 ADAPT point from arXiv:2603.13073v2 Appendix C: n_ADAPT=324.5 +/- 2.1.",
    },
}


TABLE_FIELDNAMES = [
    "case_id",
    "source_name",
    "h0.25",
    "adapt_chem_accuracy_iteration",
    "has_adapt_chem_hit",
    "adapt_source",
    "adapt_note",
    "sweep_best_parameter_name",
    "sweep_best_parameter_value",
    "sweep_best_iteration",
    "sweep_converged_values_iterations",
    "method",
    "method_chem_accuracy_iteration",
    "method_hit_chemical_accuracy",
    "method_status",
    "casscf_energy_hartree",
]

SWEEP_FIELDNAMES = [
    "case_id",
    "source_name",
    "h0.25",
    "method",
    "parameter_name",
    "parameter_value",
    "hit_iteration",
    "hit_chemical_accuracy",
    "status",
    "n_steps_requested",
    "n_steps_completed",
    "stopped_early",
    "final_min_history_energy_hartree",
    "casscf_energy_hartree",
    "summary_path",
]


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _adapt_override(case_id: str) -> dict[str, Any] | None:
    summary_path = ADAPT_OVERRIDE_SUMMARIES.get(case_id)
    if summary_path is not None:
        summary = _load_summary(summary_path)
        if (
            summary is not None
            and summary.get("status") == "completed"
            and bool(summary.get("hit_chemical_accuracy", False))
            and summary.get("hit_iteration") is not None
        ):
            return {
                "iteration": float(summary["hit_iteration"]),
                "source": "qcant_adapt_summary",
                "note": str(summary_path.relative_to(BENCHMARK_ROOT)),
            }
    return ADAPT_FIXED_OVERRIDES.get(case_id)


def _number_or_blank(value: str | int | float | None) -> float | int | str:
    if value is None or value == "":
        return ""
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _build_sweep_rows(method: str, manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    manifest_by_case = {row["case_id"]: row for row in manifest_rows}
    sweep_root = SWEEP_DIRS[method]
    if not sweep_root.exists():
        return []
    parameter_name = SWEEP_PARAMETER_NAMES[method]
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(sweep_root.glob("*/*/summary.json")):
        case_id = summary_path.parent.name
        manifest_row = manifest_by_case.get(case_id)
        if manifest_row is None:
            continue
        summary = _load_summary(summary_path)
        if summary is None:
            continue
        run_config = _load_summary(summary_path.parent / "run_config.json") or {}
        parameter_value = run_config.get(parameter_name)
        if parameter_value is None:
            continue
        if round(float(parameter_value), 10) not in CANONICAL_SWEEP_VALUE_SET:
            continue
        hit_iteration = summary.get("hit_iteration")
        rows.append(
            {
                "case_id": case_id,
                "source_name": manifest_row["source_name"],
                "h0.25": float(manifest_row["h0.25"]),
                "method": method,
                "parameter_name": parameter_name,
                "parameter_value": float(parameter_value),
                "hit_iteration": int(hit_iteration) if hit_iteration is not None else "",
                "hit_chemical_accuracy": str(bool(summary.get("hit_chemical_accuracy", False))).lower(),
                "status": str(summary.get("status", "unknown")),
                "n_steps_requested": int(summary.get("n_steps_requested", run_config.get("n_steps", 0)) or 0),
                "n_steps_completed": int(summary.get("n_steps_completed", summary.get("n_steps", 0)) or 0),
                "stopped_early": str(bool(summary.get("stopped_early", False))).lower(),
                "final_min_history_energy_hartree": summary.get("final_min_history_energy_hartree", ""),
                "casscf_energy_hartree": float(manifest_row["casscf_energy_hartree"]),
                "summary_path": str(summary_path.relative_to(BENCHMARK_ROOT)),
            }
        )
    return rows


def _sweep_summary_by_case(sweep_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    hits_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in sweep_rows:
        if row["status"] != "completed" or row["hit_chemical_accuracy"] != "true" or row["hit_iteration"] == "":
            continue
        hits_by_case.setdefault(str(row["case_id"]), []).append(row)

    summary: dict[str, dict[str, str]] = {}
    for case_id, rows in hits_by_case.items():
        rows = sorted(rows, key=lambda item: (int(item["hit_iteration"]), float(item["parameter_value"])))
        best = rows[0]
        values_iterations = [
            f"{float(row['parameter_value']):g}:{int(row['hit_iteration'])}" for row in sorted(rows, key=lambda item: float(item["parameter_value"]))
        ]
        summary[case_id] = {
            "sweep_best_parameter_name": str(best["parameter_name"]),
            "sweep_best_parameter_value": f"{float(best['parameter_value']):g}",
            "sweep_best_iteration": str(int(best["hit_iteration"])),
            "sweep_converged_values_iterations": ";".join(values_iterations),
        }
    return summary


def _build_table(
    method: str, manifest_rows: list[dict[str, str]], run_dir: Path, sweep_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sweep_summary = _sweep_summary_by_case(sweep_rows)
    rows = []
    for row in manifest_rows:
        adapt_iteration = _number_or_blank(row["adapt_chem_accuracy_iteration"])
        has_adapt_hit = row["has_adapt_chem_hit"]
        adapt_source = "manifest" if has_adapt_hit == "true" else ""
        adapt_note = ""
        override = _adapt_override(row["case_id"])
        if override is not None:
            adapt_iteration = _number_or_blank(override["iteration"])
            has_adapt_hit = "true"
            adapt_source = str(override["source"])
            adapt_note = str(override["note"])

        summary = _load_summary(run_dir / row["case_id"] / "summary.json")
        method_hit = ""
        method_iter = ""
        method_status = "missing"
        if summary is not None:
            method_status = str(summary.get("status", "unknown"))
            method_hit = str(bool(summary.get("hit_chemical_accuracy", False))).lower()
            if summary.get("hit_iteration") is not None:
                method_iter = int(summary["hit_iteration"])

        rows.append(
            {
                "case_id": row["case_id"],
                "source_name": row["source_name"],
                "h0.25": float(row["h0.25"]),
                "adapt_chem_accuracy_iteration": adapt_iteration,
                "has_adapt_chem_hit": has_adapt_hit,
                "adapt_source": adapt_source,
                "adapt_note": adapt_note,
                "sweep_best_parameter_name": sweep_summary.get(row["case_id"], {}).get("sweep_best_parameter_name", ""),
                "sweep_best_parameter_value": sweep_summary.get(row["case_id"], {}).get("sweep_best_parameter_value", ""),
                "sweep_best_iteration": sweep_summary.get(row["case_id"], {}).get("sweep_best_iteration", ""),
                "sweep_converged_values_iterations": sweep_summary.get(row["case_id"], {}).get(
                    "sweep_converged_values_iterations", ""
                ),
                "method": method,
                "method_chem_accuracy_iteration": method_iter,
                "method_hit_chemical_accuracy": method_hit,
                "method_status": method_status,
                "casscf_energy_hartree": float(row["casscf_energy_hartree"]),
            }
        )
    return rows


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_sweep_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _read_table(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_xy(
    rows: list[dict[str, Any]], y_key: str, row_filter: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row_filter is not None and not row_filter(row):
            continue
        if row["has_adapt_chem_hit"] != "true":
            continue
        value = row[y_key]
        if value == "" or value is None:
            continue
        y = float(value)
        if y > 0.0 and math.isfinite(y):
            points.append((float(row["h0.25"]), y))
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    points.sort(key=lambda item: item[0])
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _finite_sweep_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    for row in rows:
        if row["status"] != "completed" or row["hit_chemical_accuracy"] != "true":
            continue
        value = row["hit_iteration"]
        if value == "" or value is None:
            continue
        y = float(value)
        parameter_value = float(row["parameter_value"])
        if y > 0.0 and math.isfinite(y) and math.isfinite(parameter_value):
            points.append((float(row["h0.25"]), y, parameter_value))
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    points.sort(key=lambda item: (item[0], item[2], item[1]))
    x, y, parameter_values = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(parameter_values, dtype=float)


def _fit_semilog(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    if x.size < 2:
        return None
    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    x_fit = np.linspace(float(x.min()), float(x.max()), 200)
    y_fit = np.exp(slope * x_fit + intercept)
    predicted = slope * x + intercept
    ss_res = float(np.sum((log_y - predicted) ** 2))
    ss_tot = float(np.sum((log_y - float(np.mean(log_y))) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return x_fit, y_fit, float(r2)


def _plot(rows: list[dict[str, Any]], sweep_rows: list[dict[str, Any]], method: str, png_path: Path, pdf_path: Path) -> None:
    adapt_x, adapt_y = _finite_xy(rows, "adapt_chem_accuracy_iteration")
    method_x, method_y = _finite_xy(rows, "method_chem_accuracy_iteration")
    sweep_x, sweep_y, sweep_parameter_values = _finite_sweep_xy(sweep_rows)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(adapt_x, adapt_y, s=48, color="#0072B2", marker="o", label="ADAPT")
    if method_x.size:
        ax.scatter(
            method_x,
            method_y,
            s=58,
            color="#D55E00",
            marker="s" if method == "qrte" else "^",
            label=method.upper(),
        )
    if sweep_x.size:
        scatter = ax.scatter(
            sweep_x,
            sweep_y,
            s=64,
            c=sweep_parameter_values,
            cmap="Oranges" if method == "qrte" else "Greens",
            vmin=0.1,
            vmax=10.0,
            marker="D",
            edgecolor="black",
            linewidth=0.25,
            label=f"{method.upper()} sweep hits",
            zorder=3,
        )
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
        colorbar.set_label(SWEEP_PARAMETER_NAMES[method])

    adapt_fit = _fit_semilog(adapt_x, adapt_y)
    if adapt_fit is not None:
        x_fit, y_fit, r2 = adapt_fit
        ax.plot(x_fit, y_fit, color="#0072B2", linewidth=1.8, alpha=0.8, label=f"ADAPT fit R2={r2:.3f}")

    method_fit = _fit_semilog(method_x, method_y)
    if method_fit is not None:
        x_fit, y_fit, r2 = method_fit
        ax.plot(
            x_fit,
            y_fit,
            color="#D55E00",
            linewidth=1.8,
            alpha=0.8,
            label=f"{method.upper()} fit R2={r2:.3f}",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Renyi entropy h0.25")
    ax.set_ylabel("Iterations to chemical accuracy")
    ax.set_title(f"ADAPT vs {method.upper()} on MolVQE-21")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _collect_one(method: str, manifest: Path, plot_dir: Path) -> None:
    manifest_rows = _load_manifest(manifest)
    sweep_rows = _build_sweep_rows(method, manifest_rows)
    rows = _build_table(method, manifest_rows, RUN_DIRS[method], sweep_rows)
    prefix = OUTPUT_PREFIXES[method]
    csv_path = plot_dir / f"{prefix}.csv"
    sweep_csv_path = plot_dir / f"{prefix}_sweep_points.csv"
    png_path = plot_dir / f"{prefix}.png"
    pdf_path = plot_dir / f"{prefix}.pdf"
    _write_table(rows, csv_path)
    _write_sweep_table(sweep_rows, sweep_csv_path)
    _plot(rows, sweep_rows, method, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {sweep_csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def _plot_one_from_csv(method: str, plot_dir: Path) -> None:
    prefix = OUTPUT_PREFIXES[method]
    csv_path = plot_dir / f"{prefix}.csv"
    sweep_csv_path = plot_dir / f"{prefix}_sweep_points.csv"
    png_path = plot_dir / f"{prefix}.png"
    pdf_path = plot_dir / f"{prefix}.pdf"
    rows = _read_table(csv_path)
    sweep_rows = _read_table(sweep_csv_path) if sweep_csv_path.exists() else []
    _plot(rows, sweep_rows, method, png_path, pdf_path)
    print(f"Read {csv_path}")
    if sweep_csv_path.exists():
        print(f"Read {sweep_csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud", "both"), default="both")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--from-csv", action="store_true", help="Regenerate plots from existing plot CSV files.")
    args = parser.parse_args()

    methods = ("qrte", "qkud") if args.method == "both" else (args.method,)
    for method in methods:
        if args.from_csv:
            _plot_one_from_csv(method, args.plot_dir)
        else:
            _collect_one(method, args.manifest, args.plot_dir)


if __name__ == "__main__":
    main()
