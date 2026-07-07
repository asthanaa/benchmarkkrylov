#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot FeNO2+ sfX2C1eROHF QRTE convergence for dt=0.01..0.10."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(TMPDIR / "python-pycache-benchmarkkrylov"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import ScalarFormatter


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = BENCHMARK_ROOT / "runs" / "qrte_sfx2c1erohf_natorb_feno_dt0.01_to_0.10_100iter"
DEFAULT_OUTPUT_PREFIX = (
    BENCHMARK_ROOT
    / "plots"
    / "feno_2plus_qrte_sfx2c1erohf_dt0p01_to_0p10_100iter_convergence"
)
CHEMICAL_ACCURACY_HARTREE = 1.6e-3


def _dt_label(delta_t: float) -> str:
    return f"dt{delta_t:.2f}".replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    iterations: list[int] = []
    errors: list[float] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            iterations.append(int(row["iteration"]))
            errors.append(float(row["error_vs_local_reference_hartree"]))
    return np.asarray(iterations, dtype=int), np.asarray(errors, dtype=float)


def _collect(run_root: Path, delta_ts: list[float]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    missing: list[str] = []
    for delta_t in delta_ts:
        run_dir = run_root / _dt_label(delta_t) / "feno_2plus"
        summary_path = run_dir / "summary.json"
        history_path = run_dir / "min_energy_history.csv"
        if not summary_path.exists() or not history_path.exists():
            missing.append(str(run_dir.relative_to(BENCHMARK_ROOT)))
            continue
        summary = _load_json(summary_path)
        if summary.get("status") != "completed":
            missing.append(str(run_dir.relative_to(BENCHMARK_ROOT)))
            continue
        iterations, errors = _read_history(history_path)
        if iterations.size == 0:
            missing.append(str(run_dir.relative_to(BENCHMARK_ROOT)))
            continue
        series.append(
            {
                "delta_t": float(delta_t),
                "iterations": iterations,
                "errors": errors,
                "summary": summary,
                "summary_path": summary_path,
            }
        )
    if missing:
        raise SystemExit("Missing or invalid dt runs:\n  " + "\n  ".join(missing))
    return series


def _write_summary_csv(series: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "delta_t",
        "n_steps_completed",
        "hit_chemical_accuracy",
        "hit_iteration",
        "final_error_hartree",
        "min_error_hartree",
        "local_reference_energy_hartree",
        "hamiltonian_cache_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in series:
            errors = np.asarray(item["errors"], dtype=float)
            summary = item["summary"]
            writer.writerow(
                {
                    "delta_t": item["delta_t"],
                    "n_steps_completed": int(summary["n_steps_completed"]),
                    "hit_chemical_accuracy": bool(summary["hit_chemical_accuracy"]),
                    "hit_iteration": "" if summary["hit_iteration"] is None else int(summary["hit_iteration"]),
                    "final_error_hartree": float(errors[-1]),
                    "min_error_hartree": float(np.min(errors)),
                    "local_reference_energy_hartree": float(summary["local_reference_energy_hartree"]),
                    "hamiltonian_cache_path": summary["hamiltonian_cache_path"],
                }
            )


def _plot(series: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.3,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "xtick.major.size": 3.4,
            "ytick.major.size": 3.4,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
        }
    )

    delta_values = np.asarray([item["delta_t"] for item in series], dtype=float)
    norm = Normalize(vmin=float(delta_values.min()), vmax=float(delta_values.max()))
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(6.7, 4.25), constrained_layout=True)
    for item in series:
        delta_t = float(item["delta_t"])
        ax.semilogy(
            item["iterations"],
            item["errors"],
            color=cmap(norm(delta_t)),
            linewidth=1.75,
            marker="o",
            markersize=2.5,
            markevery=5,
            label=fr"$\Delta t={delta_t:.2f}$",
        )

    ax.axhline(
        CHEMICAL_ACCURACY_HARTREE,
        color="#C43B3B",
        linestyle="--",
        linewidth=1.25,
        label=f"chemical accuracy ({CHEMICAL_ACCURACY_HARTREE:g} Ha)",
    )
    ax.set_xlabel("QRTE iteration")
    ax.set_ylabel("Error vs local reference (Ha)")
    ax.set_title(r"$\mathrm{[FeNO]^{2+}}$ sfX2C1eROHF QRTE convergence")
    ax.grid(True, which="major", color="#D6D6D6", alpha=0.50, linewidth=0.55)
    ax.grid(True, which="minor", color="#ECECEC", alpha=0.30, linewidth=0.35)
    ax.tick_params(which="both", direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 101)

    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.015)
    colorbar.set_label(r"$\Delta t$")
    colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())

    reference_energy = float(series[0]["summary"]["local_reference_energy_hartree"])
    ax.text(
        0.03,
        0.05,
        f"reference = {reference_energy:.12f} Ha\n100 QRTE iterations",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        bbox=dict(facecolor="white", edgecolor="#D4D4D4", alpha=0.86, boxstyle="round,pad=0.3"),
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delta_ts = [index / 100.0 for index in range(1, 11)]
    series = _collect(args.run_root, delta_ts)
    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    _write_summary_csv(series, csv_path)
    _plot(series, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    for item in series:
        summary = item["summary"]
        errors = np.asarray(item["errors"], dtype=float)
        print(
            f"dt={item['delta_t']:.2f}: hit={summary['hit_chemical_accuracy']} "
            f"final_error={errors[-1]:.6g} min_error={np.min(errors):.6g}"
        )


if __name__ == "__main__":
    main()
