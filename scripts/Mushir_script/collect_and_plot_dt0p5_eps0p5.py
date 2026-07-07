#!/usr/bin/env python
"""Collect and plot QRTE/QKUD 0.5 benchmark summaries.

This mirrors scripts/collect_and_plot.py, but it reads the dedicated 0.5 run
directories instead of the production 0.1 directories:

    runs/qrte_natorb_dt0.5
    runs/qkud_natorb_eps0.5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))

import collect_and_plot as base_plot  # noqa: E402


DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_PLOT_DIR = BENCHMARK_ROOT / "plots" / "mushir_dt0p5_eps0p5"
RUN_DIRS = {
    "qrte": BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.5",
    "qkud": BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.5",
}
OUTPUT_PREFIXES = {
    "qrte": "adapt_vs_qrte_dt0p5_iterations_vs_entropy",
    "qkud": "adapt_vs_qkud_eps0p5_iterations_vs_entropy",
}
PLOT_TITLES = {
    "qrte": "ADAPT vs QRTE on MolVQE-21, delta_t=0.5",
    "qkud": "ADAPT vs QKUD on MolVQE-21, epsilon=0.5",
}


def _write_empty_sweep_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_plot.SWEEP_FIELDNAMES, lineterminator="\n")
        writer.writeheader()


def _write_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_plot.TABLE_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], method: str, png_path: Path, pdf_path: Path) -> None:
    adapt_x, adapt_y = base_plot._finite_xy(rows, "adapt_chem_accuracy_iteration")
    method_x, method_y = base_plot._finite_xy(rows, "method_chem_accuracy_iteration")

    base_plot.plt.rcParams.update(
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

    fig, ax = base_plot.plt.subplots(figsize=(7.2, 4.8))
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

    adapt_fit = base_plot._fit_semilog(adapt_x, adapt_y)
    if adapt_fit is not None:
        x_fit, y_fit, r2 = adapt_fit
        ax.plot(x_fit, y_fit, color="#0072B2", linewidth=1.8, alpha=0.8, label=f"ADAPT fit R2={r2:.3f}")

    method_fit = base_plot._fit_semilog(method_x, method_y)
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
    ax.set_title(PLOT_TITLES[method])
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    base_plot.plt.close(fig)


def _collect_one(method: str, manifest: Path, plot_dir: Path) -> None:
    manifest_rows = base_plot._load_manifest(manifest)
    rows = base_plot._build_table(method, manifest_rows, RUN_DIRS[method], [])
    prefix = OUTPUT_PREFIXES[method]
    csv_path = plot_dir / f"{prefix}.csv"
    sweep_csv_path = plot_dir / f"{prefix}_sweep_points.csv"
    png_path = plot_dir / f"{prefix}.png"
    pdf_path = plot_dir / f"{prefix}.pdf"
    _write_table(rows, csv_path)
    _write_empty_sweep_table(sweep_csv_path)
    _plot(rows, method, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {sweep_csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def _plot_one_from_csv(method: str, plot_dir: Path) -> None:
    prefix = OUTPUT_PREFIXES[method]
    csv_path = plot_dir / f"{prefix}.csv"
    png_path = plot_dir / f"{prefix}.png"
    pdf_path = plot_dir / f"{prefix}.pdf"
    rows = base_plot._read_table(csv_path)
    _plot(rows, method, png_path, pdf_path)
    print(f"Read {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("qrte", "qkud", "both"), default="both")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--from-csv", action="store_true", help="Regenerate plots from existing 0.5 plot CSV files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    methods = ("qrte", "qkud") if args.method == "both" else (args.method,)
    for method in methods:
        if args.from_csv:
            _plot_one_from_csv(method, args.plot_dir)
        else:
            _collect_one(method, args.manifest, args.plot_dir)


if __name__ == "__main__":
    main()
