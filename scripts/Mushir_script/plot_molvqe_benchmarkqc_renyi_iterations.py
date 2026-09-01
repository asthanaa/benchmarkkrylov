#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot MolVQE-21 and BenchmarkQC Krylov hit iterations versus Renyi entropy."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(TMPDIR / f"matplotlib-{os.environ.get('USER', 'benchmarkkrylov')}"),
)
os.environ.setdefault(
    "PYTHONPYCACHEPREFIX",
    str(TMPDIR / f"python-pycache-{os.environ.get('USER', 'benchmarkkrylov')}"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    BENCHMARK_ROOT
    / "plots"
    / "molvqe_benchmarkqc_qrte_qkud_iterations_vs_cumulant_norm2.csv"
)
DEFAULT_OUTPUT_PREFIX = (
    BENCHMARK_ROOT
    / "plots"
    / "molvqe_benchmarkqc_qrte_qkud_iterations_vs_renyi_entropy"
)

FIELDNAMES = [
    "dataset",
    "case_id",
    "renyi_0p25_nats",
    "qrte_hit_chemical_accuracy",
    "qrte_chem_accuracy_iteration",
    "qrte_status",
    "qrte_n_steps_completed",
    "qrte_n_steps_requested",
    "qrte_summary_path",
    "qkud_hit_chemical_accuracy",
    "qkud_chem_accuracy_iteration",
    "qkud_status",
    "qkud_n_steps_completed",
    "qkud_n_steps_requested",
    "qkud_summary_path",
    "active_electrons",
    "active_orbitals",
    "spin",
    "n_alpha",
    "n_beta",
    "sector_dim",
]


def _positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if math.isfinite(number) and number > 0.0:
        return number
    return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for source_row in csv.DictReader(handle):
            entropy = _positive_float(source_row.get("h0.25"))
            if entropy is None:
                raise ValueError(
                    f"{source_row.get('case_id', '<unknown>')}: missing finite positive h0.25"
                )
            row = {field: source_row.get(field, "") for field in FIELDNAMES}
            row["renyi_0p25_nats"] = entropy
            rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row["renyi_0p25_nats"]),
            str(row["dataset"]),
            str(row["case_id"]),
        )
    )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDNAMES})
    tmp_path.replace(path)


def _hit_values(
    rows: list[dict[str, Any]],
    *,
    method: str,
    dataset: str,
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        if row[f"{method}_hit_chemical_accuracy"] != "true":
            continue
        iteration = _positive_float(row[f"{method}_chem_accuracy_iteration"])
        if iteration is None:
            continue
        points.append((float(row["renyi_0p25_nats"]), iteration))
    points.sort(key=lambda point: point[0])
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


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


def _plot(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
            "axes.labelsize": 13.5,
            "axes.titlesize": 14.5,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "legend.fontsize": 10.5,
        }
    )

    method_specs = [
        ("qrte", "QRTE dt=0.05", "#1F77B4"),
        ("qkud", "QKUD eps=0.05", "#D62728"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=True)
    for ax, (method, title, color) in zip(axes, method_specs, strict=True):
        for dataset, marker, face, edge, size, linewidth in [
            ("MolVQE-21", "o", color, "white", 58, 0.6),
            ("BenchmarkQC", "D", "white", color, 64, 1.55),
        ]:
            x, y = _hit_values(rows, method=method, dataset=dataset)
            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=0.92,
                label=f"{dataset} (n={len(y)})",
                zorder=5,
            )

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        for dataset in ("MolVQE-21", "BenchmarkQC"):
            x, y = _hit_values(rows, method=method, dataset=dataset)
            if x.size:
                x_parts.append(x)
                y_parts.append(y)
        if x_parts:
            fit = _fit_log_y(np.concatenate(x_parts), np.concatenate(y_parts))
            if fit is not None:
                x_fit, y_fit, r2 = fit
                ax.plot(
                    x_fit,
                    y_fit,
                    color=color,
                    linewidth=1.8,
                    alpha=0.58,
                    label=rf"hit fit $R^2$={r2:.3f}",
                    zorder=3,
                )

        ax.set_title(title)
        ax.set_xlabel(r"Renyi entropy, $H_{0.25}$ (nats)")
        ax.set_yscale("log")
        ax.set_xlim(left=0.0)
        ax.grid(True, which="major", color="#DADADA", alpha=0.65, linewidth=0.75)
        ax.grid(True, which="minor", color="#ECECEC", alpha=0.48, linewidth=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="#D0D0D0",
            framealpha=0.94,
            borderpad=0.42,
            handlelength=1.3,
            labelspacing=0.42,
        )

    axes[0].set_ylabel("Iterations to chemical accuracy")
    fig.suptitle(
        "MolVQE-21 and BenchmarkQC Krylov convergence vs Renyi entropy",
        fontsize=15.5,
        y=0.985,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.13, wspace=0.12)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_csv)
    dataset_counts = {
        dataset: sum(row["dataset"] == dataset for row in rows)
        for dataset in ("MolVQE-21", "BenchmarkQC")
    }
    if dataset_counts != {"MolVQE-21": 27, "BenchmarkQC": 51}:
        raise SystemExit(f"Expected 27 MolVQE-21 and 51 BenchmarkQC rows, found {dataset_counts}")

    csv_path = args.output_prefix.with_suffix(".csv")
    _write_csv(rows, csv_path)
    _plot(rows, args.output_prefix)

    for method in ("qrte", "qkud"):
        hit_count = sum(row[f"{method}_hit_chemical_accuracy"] == "true" for row in rows)
        miss_count = len(rows) - hit_count
        print(f"{method}: {hit_count} hits plotted; {miss_count} no-hit rows excluded")
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
