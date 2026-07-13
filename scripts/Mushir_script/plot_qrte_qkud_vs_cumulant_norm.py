#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot QRTE/QKUD iterations to chemical accuracy versus two-body cumulant norm."""

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
DEFAULT_INPUT_CSV = (
    BENCHMARK_ROOT
    / "runs"
    / "two_body_cumulant_norms"
    / "qkud_eps0p05_converged"
    / "two_body_cumulant_norms.csv"
)
DEFAULT_OUTPUT_PREFIX = (
    BENCHMARK_ROOT / "plots" / "qrte_qkud_iterations_vs_2body_cumulant_norm2"
)
DEFAULT_ITERATIONS_CSV = (
    BENCHMARK_ROOT / "plots" / "qkud_eps0p05_vs_qrte_dt0p05_iterations_vs_entropy.csv"
)
DEFAULT_BENCHMARK_QC_CSV = (
    BENCHMARK_ROOT / "runs" / "benchmark_qc_2body_cumulants" / "two_body_cumulants.csv"
)
DEFAULT_BENCHMARK_QC_QKUD_DIR = BENCHMARK_ROOT / "runs" / "benchmark_qc_qkud_eps0.05"
DEFAULT_BENCHMARK_QC_QRTE_DIR = BENCHMARK_ROOT / "runs" / "benchmark_qc_qrte_dt0.05"
BASE_EXPECTED_QKUD_HITS = 27
BASE_EXPECTED_QRTE_HITS = 25
EXCLUDED_CASE_IDS = {"U2_eq", "FeS_eq", "N2_eq"}

FIELDNAMES = [
    "case_id",
    "cumulant_2rdm_frobenius_norm_squared",
    "cumulant_2rdm_frobenius_norm",
    "adapt_chem_accuracy_iteration",
    "qkud_eps0p05_chem_accuracy_iteration",
    "qrte_dt0p05_chem_accuracy_iteration",
    "qkud_only_hit",
    "h0.25",
    "n_spatial_orbitals",
    "n_alpha",
    "n_beta",
    "sector_dim",
]


def _read_adapt_iterations(path: Path) -> dict[str, float]:
    adapt_by_case: dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("adapt_hit_chemical_accuracy") != "true":
                continue
            value = _positive_number(row.get("adapt_chem_accuracy_iteration"))
            if value is not None:
                adapt_by_case[str(row["case_id"])] = value
    return adapt_by_case


def _read_rows(path: Path, iterations_csv: Path) -> list[dict[str, Any]]:
    adapt_by_case = _read_adapt_iterations(iterations_csv)
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            norm2 = _positive_number(row.get("cumulant_2rdm_frobenius_norm_squared"))
            qkud_iter = _positive_number(row.get("qkud_eps0p05_chem_accuracy_iteration"))
            if norm2 is None or qkud_iter is None:
                continue
            rows.append(
                {
                    **row,
                    "cumulant_2rdm_frobenius_norm_squared": norm2,
                    "cumulant_2rdm_frobenius_norm": _positive_number(
                        row.get("cumulant_2rdm_frobenius_norm")
                    ),
                    "adapt_chem_accuracy_iteration": adapt_by_case.get(str(row["case_id"])),
                    "qkud_eps0p05_chem_accuracy_iteration": qkud_iter,
                    "qrte_dt0p05_chem_accuracy_iteration": _positive_number(
                        row.get("qrte_dt0p05_chem_accuracy_iteration")
                    ),
                    "h0.25": _positive_number(row.get("h0.25")),
                    "n_spatial_orbitals": int(row["n_spatial_orbitals"]),
                    "n_alpha": int(row["n_alpha"]),
                    "n_beta": int(row["n_beta"]),
                    "sector_dim": int(row["sector_dim"]),
                }
            )
    rows.sort(key=lambda item: (float(item["cumulant_2rdm_frobenius_norm_squared"]), item["case_id"]))
    return rows


def _read_hit_iteration(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if str(summary.get("hit_chemical_accuracy", "")).lower() != "true":
        return None
    return _positive_number(summary.get("hit_iteration"))


def _read_benchmark_qc_rows(
    cumulants_csv: Path, qkud_dir: Path, qrte_dir: Path
) -> list[dict[str, Any]]:
    if not cumulants_csv.exists():
        return []

    rows: list[dict[str, Any]] = []
    with cumulants_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status", "")).lower() != "completed":
                continue
            norm2 = _positive_number(row.get("cumulant_2rdm_frobenius_norm_squared"))
            case_id = str(row["case_id"])
            qkud_iter = _read_hit_iteration(qkud_dir / case_id / "summary.json")
            if norm2 is None or qkud_iter is None:
                continue
            qrte_iter = _read_hit_iteration(qrte_dir / case_id / "summary.json")
            rows.append(
                {
                    "case_id": case_id,
                    "cumulant_2rdm_frobenius_norm_squared": norm2,
                    "cumulant_2rdm_frobenius_norm": _positive_number(
                        row.get("cumulant_2rdm_frobenius_norm")
                    ),
                    "adapt_chem_accuracy_iteration": None,
                    "qkud_eps0p05_chem_accuracy_iteration": qkud_iter,
                    "qrte_dt0p05_chem_accuracy_iteration": qrte_iter,
                    "qkud_only_hit": str(qrte_iter is None).lower(),
                    "h0.25": None,
                    "n_spatial_orbitals": int(row["active_orbitals"]),
                    "n_alpha": int(row["n_alpha"]),
                    "n_beta": int(row["n_beta"]),
                    "sector_dim": int(row["sector_dim"]),
                    "benchmark_qc": True,
                }
            )
    rows.sort(key=lambda item: (float(item["cumulant_2rdm_frobenius_norm_squared"]), item["case_id"]))
    return rows


def _positive_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number > 0.0 and math.isfinite(number):
        return number
    return None


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def _xy(rows: list[dict[str, Any]], y_key: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    selected = [row for row in rows if _positive_number(row.get(y_key)) is not None]
    x = np.asarray([float(row["cumulant_2rdm_frobenius_norm_squared"]) for row in selected], dtype=float)
    y = np.asarray([float(row[y_key]) for row in selected], dtype=float)
    return x, y, selected


def _fit_log_y(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
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


def _annotate(ax, row: dict[str, Any], y_key: str, *, xytext: tuple[int, int]) -> None:
    y = _positive_number(row.get(y_key))
    if y is None:
        return
    ax.annotate(
        str(row["case_id"]),
        (float(row["cumulant_2rdm_frobenius_norm_squared"]), y),
        xytext=xytext,
        textcoords="offset points",
        fontsize=8.0,
        color="#333333",
        ha="left",
        va="bottom",
    )


def _plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
    qrte_x, qrte_y, qrte_rows = _xy(rows, "qrte_dt0p05_chem_accuracy_iteration")
    qkud_x, qkud_y, qkud_rows = _xy(rows, "qkud_eps0p05_chem_accuracy_iteration")
    benchmark_rows = [row for row in rows if row.get("benchmark_qc")]
    special_case_ids = {"FeO", "feno_2plus"}
    qkud_special_rows = [
        row
        for row in qkud_rows
        if str(row.get("case_id", "")) in special_case_ids
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.75,
            "ytick.minor.width": 0.75,
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.minor.size": 2.6,
            "ytick.minor.size": 2.6,
        }
    )

    qrte_color = "#1F77B4"
    qkud_color = "#D62728"

    fig, ax = plt.subplots(figsize=(8.8, 5.9), constrained_layout=False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.975, bottom=0.225)
    ax.scatter(
        qrte_x,
        qrte_y,
        s=58,
        color=qrte_color,
        marker="s",
        label=rf"QRTE $dt=0.05$ (n={len(qrte_rows)})",
        zorder=4,
    )
    ax.scatter(
        qkud_x,
        qkud_y,
        s=62,
        color=qkud_color,
        marker="^",
        label=rf"QKUD $\epsilon=0.05$ (n={len(qkud_rows)})",
        zorder=5,
    )

    if qkud_special_rows:
        ax.scatter(
            [float(row["cumulant_2rdm_frobenius_norm_squared"]) for row in qkud_special_rows],
            [float(row["qkud_eps0p05_chem_accuracy_iteration"]) for row in qkud_special_rows],
            s=190,
            facecolor="#F0C808",
            edgecolor="#111111",
            linewidth=0.8,
            marker="*",
            label="_nolegend_",
            zorder=6,
        )

    if benchmark_rows:
        benchmark_qrte_x, benchmark_qrte_y, _ = _xy(
            benchmark_rows, "qrte_dt0p05_chem_accuracy_iteration"
        )
        benchmark_qkud_x, benchmark_qkud_y, _ = _xy(
            benchmark_rows, "qkud_eps0p05_chem_accuracy_iteration"
        )
        ax.scatter(
            benchmark_qrte_x,
            benchmark_qrte_y,
            s=115,
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.05,
            marker="s",
            label="_nolegend_",
            zorder=7,
        )
        ax.scatter(
            benchmark_qkud_x,
            benchmark_qkud_y,
            s=120,
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.05,
            marker="^",
            label="_nolegend_",
            zorder=8,
        )
        for row in benchmark_rows:
            _annotate(
                ax,
                row,
                "qrte_dt0p05_chem_accuracy_iteration",
                xytext=(5, 5),
            )

    for row in rows:
        qkud_iter = _positive_number(row.get("qkud_eps0p05_chem_accuracy_iteration"))
        qrte_iter = _positive_number(row.get("qrte_dt0p05_chem_accuracy_iteration"))
        if qkud_iter is None or qrte_iter is None:
            continue
        x = float(row["cumulant_2rdm_frobenius_norm_squared"])
        ax.plot([x, x], [qkud_iter, qrte_iter], color="#999999", alpha=0.25, linewidth=0.8, zorder=1)

    for label, x, y, color, linestyle in (
        (r"QRTE $dt=0.05$", qrte_x, qrte_y, qrte_color, "-"),
        (r"QKUD $\epsilon=0.05$", qkud_x, qkud_y, qkud_color, "-"),
    ):
        fit = _fit_log_y(x, y)
        if fit is None:
            continue
        x_fit, y_fit, r2 = fit
        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle=linestyle,
            linewidth=1.65,
            alpha=0.64,
            label=rf"{label} trend $R^2$={r2:.3f}",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\|{}^{2}\Delta\|_{\mathrm{F}}^{2}$")
    ax.set_ylabel("Iter. to chem. accuracy (~ 1 mHa)")
    ax.grid(True, which="major", color="#D9D9D9", alpha=0.5, linewidth=0.7)
    ax.grid(True, which="minor", color="#EBEBEB", alpha=0.36, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        bbox_transform=fig.transFigure,
        handlelength=1.5,
        columnspacing=1.8,
        labelspacing=0.75,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--iterations-csv", type=Path, default=DEFAULT_ITERATIONS_CSV)
    parser.add_argument("--benchmark-qc-csv", type=Path, default=DEFAULT_BENCHMARK_QC_CSV)
    parser.add_argument("--benchmark-qc-qkud-dir", type=Path, default=DEFAULT_BENCHMARK_QC_QKUD_DIR)
    parser.add_argument("--benchmark-qc-qrte-dir", type=Path, default=DEFAULT_BENCHMARK_QC_QRTE_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--expected-adapt-hits", type=int, default=27)
    parser.add_argument("--expected-qkud-hits", type=int, default=None)
    parser.add_argument("--expected-qrte-hits", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_csv, args.iterations_csv)
    seen_case_ids = {str(row["case_id"]) for row in rows}
    benchmark_qc_rows = [
        row
        for row in _read_benchmark_qc_rows(
            args.benchmark_qc_csv,
            args.benchmark_qc_qkud_dir,
            args.benchmark_qc_qrte_dir,
        )
        if str(row["case_id"]) not in seen_case_ids
    ]
    rows.extend(benchmark_qc_rows)
    rows = [row for row in rows if str(row["case_id"]) not in EXCLUDED_CASE_IDS]
    benchmark_qc_rows = [
        row for row in benchmark_qc_rows if str(row["case_id"]) not in EXCLUDED_CASE_IDS
    ]
    rows.sort(key=lambda item: (float(item["cumulant_2rdm_frobenius_norm_squared"]), item["case_id"]))
    expected_qkud_hits = (
        args.expected_qkud_hits
        if args.expected_qkud_hits is not None
        else BASE_EXPECTED_QKUD_HITS
        + sum(row["qkud_eps0p05_chem_accuracy_iteration"] is not None for row in benchmark_qc_rows)
    )
    expected_qrte_hits = (
        args.expected_qrte_hits
        if args.expected_qrte_hits is not None
        else BASE_EXPECTED_QRTE_HITS
        + sum(row["qrte_dt0p05_chem_accuracy_iteration"] is not None for row in benchmark_qc_rows)
    )
    adapt_hits = [row for row in rows if row["adapt_chem_accuracy_iteration"] is not None]
    qkud_hits = [row for row in rows if row["qkud_eps0p05_chem_accuracy_iteration"] is not None]
    qrte_hits = [row for row in rows if row["qrte_dt0p05_chem_accuracy_iteration"] is not None]
    if len(adapt_hits) != int(args.expected_adapt_hits):
        raise SystemExit(f"Expected {args.expected_adapt_hits} ADAPT hits, found {len(adapt_hits)}")
    if len(qkud_hits) != int(expected_qkud_hits):
        raise SystemExit(f"Expected {expected_qkud_hits} QKUD hits, found {len(qkud_hits)}")
    if len(qrte_hits) != int(expected_qrte_hits):
        raise SystemExit(f"Expected {expected_qrte_hits} QRTE hits, found {len(qrte_hits)}")

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    _write_rows(rows, csv_path)
    _plot(rows, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Included {len(adapt_hits)} ADAPT hits, {len(qkud_hits)} QKUD hits, and {len(qrte_hits)} QRTE hits.")


if __name__ == "__main__":
    main()
