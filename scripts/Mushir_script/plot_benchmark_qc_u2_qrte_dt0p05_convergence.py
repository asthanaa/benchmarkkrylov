#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot Benchmark-QC U2 QRTE dt=0.05 convergence against iterations."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_RUN_DIR = BENCHMARK_ROOT / "runs" / "benchmark_qc_qrte_dt0.05" / "U2_eq"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "u2_benchmark_qc_qrte_dt0p05_convergence"
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_history(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, str]]
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No convergence rows found in {path}")

    iterations = np.asarray([int(row["iteration"]) for row in rows], dtype=int)
    energies = np.asarray([float(row["min_energy_hartree"]) for row in rows], dtype=float)
    errors = np.asarray([abs(float(row["error_vs_reference_hartree"])) for row in rows], dtype=float)
    return iterations, energies, errors


def _plot(run_dir: Path, output_prefix: Path) -> tuple[Path, Path]:
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "min_energy_history.csv"
    summary = _load_json(summary_path)
    iterations, _energies, errors = _read_history(history_path)

    reference_energy = float(summary["reference_energy_hartree"])
    hit_iteration = summary.get("hit_iteration")
    final_error = float(errors[-1])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax.semilogy(
        iterations,
        errors,
        color="#0072B2",
        marker="o",
        markersize=3.4,
        linewidth=1.8,
        label=r"QRTE, $\Delta t=0.05$",
    )
    ax.axhline(
        CHEMICAL_ACCURACY_HARTREE,
        color="#C43B3B",
        linestyle="--",
        linewidth=1.25,
        label=f"chemical accuracy ({CHEMICAL_ACCURACY_HARTREE:g} Ha)",
    )
    if hit_iteration is not None:
        ax.axvline(
            int(hit_iteration),
            color="#555555",
            linestyle=":",
            linewidth=1.15,
            label=f"first hit: iteration {int(hit_iteration)}",
        )

    ax.set_xlabel("QRTE iteration")
    ax.set_ylabel("Absolute error vs Benchmark-QC reference (Ha)")
    ax.set_title(r"$\mathrm{U_2}$ Benchmark-QC QRTE convergence")
    ax.grid(True, which="major", color="#D6D6D6", alpha=0.55, linewidth=0.6)
    ax.grid(True, which="minor", color="#ECECEC", alpha=0.35, linewidth=0.4)
    ax.tick_params(which="both", direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(1, max(int(iterations[-1]), 1))
    ax.legend(frameon=False, loc="upper right")

    note = (
        f"reference = {reference_energy:.8f} Ha\n"
        f"final error = {final_error:.6g} Ha\n"
        f"iterations = {int(iterations[-1])}"
    )
    ax.text(
        0.03,
        0.05,
        note,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "#CFCFCF", "alpha": 0.92},
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    args = parser.parse_args()
    if not args.run_dir.is_absolute():
        args.run_dir = BENCHMARK_ROOT / args.run_dir
    if not args.output_prefix.is_absolute():
        args.output_prefix = BENCHMARK_ROOT / args.output_prefix
    return args


def main() -> None:
    args = parse_args()
    png_path, pdf_path = _plot(args.run_dir, args.output_prefix)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
