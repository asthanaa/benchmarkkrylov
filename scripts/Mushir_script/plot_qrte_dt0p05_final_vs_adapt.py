#!/usr/bin/env python
"""Plot final accepted QRTE dt=0.05 hits against ADAPT-VQE."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "adapt_vs_qrte_dt0p05_final_25_iterations_vs_entropy"
BASE_QRTE_DIR = BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.05"

QRTE_RERUN_SUMMARIES = {
    "cyclobutadiene_ceo": BENCHMARK_ROOT
    / "runs"
    / "qrte_corrected_active_space_dt0.05"
    / "cyclobutadiene_ceo"
    / "summary.json",
    "m_Benzyne_equi_cc_pVTZ_0": BENCHMARK_ROOT
    / "runs"
    / "qrte_mbenzyne_local_casscf_natorb_dt0.05"
    / "m_Benzyne_equi_cc_pVTZ_0"
    / "summary.json",
    "m_Benzyne_equi_cc_pVTZ_2": BENCHMARK_ROOT
    / "runs"
    / "qrte_mbenzyne_local_casscf_natorb_dt0.05"
    / "m_Benzyne_equi_cc_pVTZ_2"
    / "summary.json",
    "fenoco3_minus": BENCHMARK_ROOT
    / "runs"
    / "qrte_sfx2c1erohf_natorb_unconverged_dt0.05_100iter"
    / "fenoco3_minus"
    / "summary.json",
    "N2_equi_cc_pVTZ_2": BENCHMARK_ROOT
    / "runs"
    / "qrte_natorb_dt0.05_100iter"
    / "N2_equi_cc_pVTZ_2"
    / "summary.json",
}

ADAPT_FIXED_OVERRIDES = {
    "H10_near_equi_sto3g_0": {
        "iteration": 324.5,
        "source": "paper_fig1_extrapolated",
        "note": "H10 ADAPT point from arXiv:2603.13073v2 Appendix C: n_ADAPT=324.5 +/- 2.1.",
    },
}

FIELDNAMES = [
    "case_id",
    "source_name",
    "h0.25",
    "adapt_chem_accuracy_iteration",
    "adapt_source",
    "adapt_note",
    "qrte_dt0p05_chem_accuracy_iteration",
    "qrte_hit_chemical_accuracy",
    "qrte_status",
    "qrte_n_steps_completed",
    "qrte_n_steps_requested",
    "qrte_summary_path",
    "qrte_result_source",
    "casscf_energy_hartree",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _adapt_iteration(row: dict[str, str]) -> tuple[float | None, str, str]:
    override = ADAPT_FIXED_OVERRIDES.get(row["case_id"])
    if override is not None:
        return float(override["iteration"]), str(override["source"]), str(override["note"])

    if row.get("has_adapt_chem_hit") == "true" and row.get("adapt_chem_accuracy_iteration"):
        return float(row["adapt_chem_accuracy_iteration"]), "manifest", ""
    return None, "", ""


def _summary_for_case(case_id: str) -> tuple[Path, str, dict[str, Any] | None]:
    rerun_path = QRTE_RERUN_SUMMARIES.get(case_id)
    if rerun_path is not None:
        return rerun_path, "rerun", _load_json(rerun_path)
    base_path = BASE_QRTE_DIR / case_id / "summary.json"
    return base_path, "base", _load_json(base_path)


def _build_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        adapt_iteration, adapt_source, adapt_note = _adapt_iteration(manifest_row)
        if adapt_iteration is None:
            continue

        summary_path, result_source, summary = _summary_for_case(manifest_row["case_id"])
        if summary is None:
            continue
        if summary.get("status") != "completed" or not bool(summary.get("hit_chemical_accuracy", False)):
            continue
        hit_iteration = summary.get("hit_iteration")
        if hit_iteration is None:
            continue

        rows.append(
            {
                "case_id": manifest_row["case_id"],
                "source_name": manifest_row["source_name"],
                "h0.25": float(manifest_row["h0.25"]),
                "adapt_chem_accuracy_iteration": adapt_iteration,
                "adapt_source": adapt_source,
                "adapt_note": adapt_note,
                "qrte_dt0p05_chem_accuracy_iteration": float(hit_iteration),
                "qrte_hit_chemical_accuracy": "true",
                "qrte_status": str(summary.get("status", "")),
                "qrte_n_steps_completed": int(summary.get("n_steps_completed", summary.get("n_steps", 0)) or 0),
                "qrte_n_steps_requested": int(summary.get("n_steps_requested", 0) or 0),
                "qrte_summary_path": str(summary_path.relative_to(BENCHMARK_ROOT)),
                "qrte_result_source": result_source,
                "casscf_energy_hartree": float(manifest_row["casscf_energy_hartree"]),
            }
        )
    rows.sort(key=lambda row: float(row["h0.25"]))
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite_xy(rows: list[dict[str, Any]], y_key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["h0.25"]), float(row[y_key]))
        for row in rows
        if float(row[y_key]) > 0.0 and math.isfinite(float(row[y_key]))
    ]
    points.sort(key=lambda item: item[0])
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


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
    return x_fit, y_fit, r2


def _plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
    adapt_x, adapt_y = _finite_xy(rows, "adapt_chem_accuracy_iteration")
    qrte_x, qrte_y = _finite_xy(rows, "qrte_dt0p05_chem_accuracy_iteration")

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

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.scatter(adapt_x, adapt_y, s=50, color="#0072B2", marker="o", label="ADAPT-VQE", zorder=3)
    ax.scatter(qrte_x, qrte_y, s=60, color="#D55E00", marker="s", label="QRTE dt=0.05", zorder=3)

    for row in rows:
        x = float(row["h0.25"])
        y0 = float(row["adapt_chem_accuracy_iteration"])
        y1 = float(row["qrte_dt0p05_chem_accuracy_iteration"])
        ax.plot([x, x], [y0, y1], color="#999999", alpha=0.24, linewidth=0.8, zorder=1)

    for label, x, y, color in (
        ("ADAPT-VQE", adapt_x, adapt_y, "#0072B2"),
        ("QRTE dt=0.05", qrte_x, qrte_y, "#D55E00"),
    ):
        fit = _fit_semilog(x, y)
        if fit is None:
            continue
        x_fit, y_fit, r2 = fit
        ax.plot(x_fit, y_fit, color=color, linewidth=1.8, alpha=0.78, label=f"{label} fit R2={r2:.3f}")

    ax.set_yscale("log")
    ax.set_xlabel("Renyi entropy h0.25")
    ax.set_ylabel("Iterations to chemical accuracy")
    ax.set_title(f"ADAPT-VQE vs QRTE dt=0.05 on converged systems (n={len(rows)})")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--expected-count", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(_read_manifest(args.manifest))
    if len(rows) != int(args.expected_count):
        raise SystemExit(f"Expected {args.expected_count} converged rows, found {len(rows)}")

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    _write_rows(rows, csv_path)
    _plot(rows, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Included {len(rows)} converged QRTE dt=0.05 systems.")


if __name__ == "__main__":
    main()
