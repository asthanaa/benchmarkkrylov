#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot MolVQE-21 and BenchmarkQC Krylov hit iterations versus cumulant size."""

from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_MOLVQE_CSV = (
    BENCHMARK_ROOT / "plots" / "qrte_qkud_iterations_vs_2body_cumulant_norm2.csv"
)
DEFAULT_BENCHMARK_QC_ROOT = Path(
    "/home/mushir.kolathingal/BenchmarkQC/datasets/benchmarkQC"
)
DEFAULT_BENCHMARK_QC_QRTE_DIR = (
    BENCHMARK_ROOT / "runs" / "benchmark_qc_51_qrte_dt0.05"
)
DEFAULT_BENCHMARK_QC_QKUD_DIR = (
    BENCHMARK_ROOT / "runs" / "benchmark_qc_51_qkud_eps0.05"
)
DEFAULT_OUTPUT_PREFIX = (
    BENCHMARK_ROOT
    / "plots"
    / "molvqe_benchmarkqc_qrte_qkud_iterations_vs_cumulant_norm2"
)

FIELDNAMES = [
    "dataset",
    "case_id",
    "cumulant_2rdm_frobenius_norm_squared",
    "cumulant_2rdm_frobenius_norm",
    "benchmark_qc_cumulant_C2",
    "cumulant_source",
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
    "h0.25",
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


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _relative(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _spin_populations(active_electrons: int, spin: int) -> tuple[int, int]:
    if (active_electrons + spin) % 2 != 0:
        raise ValueError(
            f"active_electrons={active_electrons} and spin={spin} have incompatible parity"
        )
    n_alpha = (active_electrons + spin) // 2
    n_beta = active_electrons - n_alpha
    if n_alpha < 0 or n_beta < 0:
        raise ValueError(
            f"active_electrons={active_electrons} and spin={spin} imply negative population"
        )
    return n_alpha, n_beta


def _sector_dim(active_orbitals: int, n_alpha: int, n_beta: int) -> int:
    return math.comb(active_orbitals, n_alpha) * math.comb(active_orbitals, n_beta)


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "hit": False,
            "hit_iteration": None,
            "n_steps_completed": None,
            "n_steps_requested": None,
            "summary_path": path,
        }
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    hit_iteration = _positive_float(summary.get("hit_iteration"))
    hit = (
        str(summary.get("status", "")).lower() == "completed"
        and str(summary.get("hit_chemical_accuracy", "")).lower() == "true"
        and hit_iteration is not None
    )
    return {
        "status": str(summary.get("status", "")),
        "hit": hit,
        "hit_iteration": hit_iteration,
        "n_steps_completed": summary.get("n_steps_completed"),
        "n_steps_requested": summary.get("n_steps_requested"),
        "summary_path": path,
    }


def _read_molvqe_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            norm2 = _positive_float(row.get("cumulant_2rdm_frobenius_norm_squared"))
            qkud_iter = _positive_float(row.get("qkud_eps0p05_chem_accuracy_iteration"))
            if norm2 is None or qkud_iter is None:
                continue
            qrte_iter = _positive_float(row.get("qrte_dt0p05_chem_accuracy_iteration"))
            n_alpha = int(row["n_alpha"])
            n_beta = int(row["n_beta"])
            rows.append(
                {
                    "dataset": "MolVQE-21",
                    "case_id": row["case_id"],
                    "cumulant_2rdm_frobenius_norm_squared": norm2,
                    "cumulant_2rdm_frobenius_norm": _positive_float(
                        row.get("cumulant_2rdm_frobenius_norm")
                    )
                    or math.sqrt(norm2),
                    "benchmark_qc_cumulant_C2": "",
                    "cumulant_source": _relative(path),
                    "qrte_hit_chemical_accuracy": _bool_text(qrte_iter is not None),
                    "qrte_chem_accuracy_iteration": qrte_iter,
                    "qrte_status": "completed" if qrte_iter is not None else "",
                    "qrte_n_steps_completed": "",
                    "qrte_n_steps_requested": "",
                    "qrte_summary_path": "",
                    "qkud_hit_chemical_accuracy": "true",
                    "qkud_chem_accuracy_iteration": qkud_iter,
                    "qkud_status": "completed",
                    "qkud_n_steps_completed": "",
                    "qkud_n_steps_requested": "",
                    "qkud_summary_path": "",
                    "h0.25": row.get("h0.25", ""),
                    "active_electrons": n_alpha + n_beta,
                    "active_orbitals": int(row["n_spatial_orbitals"]),
                    "spin": n_alpha - n_beta,
                    "n_alpha": n_alpha,
                    "n_beta": n_beta,
                    "sector_dim": row.get("sector_dim", ""),
                }
            )
    return rows


def _read_benchmark_qc_rows(
    dataset_root: Path,
    qrte_dir: Path,
    qkud_dir: Path,
) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "source" / "manifest.csv"
    reference_path = dataset_root / "reference" / "reference_results.json"
    references = json.loads(reference_path.read_text(encoding="utf-8"))["cases"]
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for manifest_row in csv.DictReader(handle):
            case_id = manifest_row["case_id"]
            reference = references[case_id]
            cumulant_c2 = _positive_float(reference.get("cumulant_C2"))
            if cumulant_c2 is None:
                continue

            active_electrons = int(manifest_row["active_electrons"])
            active_orbitals = int(manifest_row["active_spatial_orbitals"])
            spin = int(manifest_row["spin"])
            n_alpha, n_beta = _spin_populations(active_electrons, spin)
            qrte = _read_summary(qrte_dir / case_id / "summary.json")
            qkud = _read_summary(qkud_dir / case_id / "summary.json")
            norm2 = 4.0 * cumulant_c2
            rows.append(
                {
                    "dataset": "BenchmarkQC",
                    "case_id": case_id,
                    "cumulant_2rdm_frobenius_norm_squared": norm2,
                    "cumulant_2rdm_frobenius_norm": math.sqrt(norm2),
                    "benchmark_qc_cumulant_C2": cumulant_c2,
                    "cumulant_source": _relative(reference_path),
                    "qrte_hit_chemical_accuracy": _bool_text(bool(qrte["hit"])),
                    "qrte_chem_accuracy_iteration": qrte["hit_iteration"],
                    "qrte_status": qrte["status"],
                    "qrte_n_steps_completed": qrte["n_steps_completed"],
                    "qrte_n_steps_requested": qrte["n_steps_requested"],
                    "qrte_summary_path": _relative(qrte["summary_path"]),
                    "qkud_hit_chemical_accuracy": _bool_text(bool(qkud["hit"])),
                    "qkud_chem_accuracy_iteration": qkud["hit_iteration"],
                    "qkud_status": qkud["status"],
                    "qkud_n_steps_completed": qkud["n_steps_completed"],
                    "qkud_n_steps_requested": qkud["n_steps_requested"],
                    "qkud_summary_path": _relative(qkud["summary_path"]),
                    "h0.25": reference.get("renyi_0p25_nats", ""),
                    "active_electrons": active_electrons,
                    "active_orbitals": active_orbitals,
                    "spin": spin,
                    "n_alpha": n_alpha,
                    "n_beta": n_beta,
                    "sector_dim": _sector_dim(active_orbitals, n_alpha, n_beta),
                }
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


def _method_values(
    rows: list[dict[str, Any]],
    *,
    method: str,
    dataset: str,
    hit_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        if row["dataset"] != dataset:
            continue
        hit = row[f"{method}_hit_chemical_accuracy"] == "true"
        if hit_only and not hit:
            continue
        if not hit_only and hit:
            continue
        x = _positive_float(row["cumulant_2rdm_frobenius_norm_squared"])
        y_key = f"{method}_chem_accuracy_iteration" if hit else f"{method}_n_steps_completed"
        y = _positive_float(row.get(y_key))
        if x is None or y is None:
            continue
        x_values.append(x)
        y_values.append(y)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


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
        for dataset, marker, label, face, edge, size, linewidth in [
            ("MolVQE-21", "o", "MolVQE-21", color, "white", 58, 0.6),
            ("BenchmarkQC", "D", "BenchmarkQC", "white", color, 64, 1.55),
        ]:
            x, y = _method_values(rows, method=method, dataset=dataset, hit_only=True)
            ax.scatter(
                x,
                y,
                s=size,
                marker=marker,
                facecolor=face,
                edgecolor=edge,
                linewidth=linewidth,
                alpha=0.92,
                label=f"{label} (n={len(y)})",
                zorder=5,
            )

        hit_x_parts: list[np.ndarray] = []
        hit_y_parts: list[np.ndarray] = []
        for dataset in ("MolVQE-21", "BenchmarkQC"):
            x, y = _method_values(rows, method=method, dataset=dataset, hit_only=True)
            if x.size:
                hit_x_parts.append(x)
                hit_y_parts.append(y)
        if hit_x_parts:
            hit_x = np.concatenate(hit_x_parts)
            hit_y = np.concatenate(hit_y_parts)
            fit = _fit_log_y(hit_x, hit_y)
            if fit is not None:
                x_fit, y_fit, r2 = fit
                ax.plot(
                    x_fit,
                    y_fit,
                    color=color,
                    linestyle="-",
                    linewidth=1.8,
                    alpha=0.58,
                    label=rf"hit fit $R^2$={r2:.3f}",
                    zorder=3,
                )

        ax.set_title(title)
        ax.set_xlabel(r"$\|{}^{2}\Delta\|_{\mathrm{F}}^{2}$")
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
        "MolVQE-21 and BenchmarkQC Krylov convergence vs 2-body cumulant",
        fontsize=15.5,
        y=0.985,
    )
    fig.text(
        0.5,
        0.018,
        "BenchmarkQC x-values use 4 x copied cumulant_C2 to match the MolVQE spin-orbital Frobenius norm-squared convention.",
        ha="center",
        va="bottom",
        fontsize=9.4,
        color="#404040",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.895, bottom=0.16, wspace=0.12)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molvqe-csv", type=Path, default=DEFAULT_MOLVQE_CSV)
    parser.add_argument("--benchmark-qc-root", type=Path, default=DEFAULT_BENCHMARK_QC_ROOT)
    parser.add_argument("--benchmark-qc-qrte-dir", type=Path, default=DEFAULT_BENCHMARK_QC_QRTE_DIR)
    parser.add_argument("--benchmark-qc-qkud-dir", type=Path, default=DEFAULT_BENCHMARK_QC_QKUD_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_molvqe_rows(args.molvqe_csv)
    rows.extend(
        _read_benchmark_qc_rows(
            args.benchmark_qc_root,
            args.benchmark_qc_qrte_dir,
            args.benchmark_qc_qkud_dir,
        )
    )
    rows.sort(
        key=lambda row: (
            str(row["dataset"]),
            float(row["cumulant_2rdm_frobenius_norm_squared"]),
            str(row["case_id"]),
        )
    )

    csv_path = args.output_prefix.with_suffix(".csv")
    _write_csv(rows, csv_path)
    _plot(rows, args.output_prefix)

    for method in ("qrte", "qkud"):
        hit_count = sum(row[f"{method}_hit_chemical_accuracy"] == "true" for row in rows)
        bqc_miss_count = sum(
            row["dataset"] == "BenchmarkQC"
            and row[f"{method}_status"] == "completed"
            and row[f"{method}_hit_chemical_accuracy"] != "true"
            for row in rows
        )
        print(
            f"{method}: {hit_count} hits plotted; "
            f"{bqc_miss_count} BenchmarkQC no-hit rows excluded from figure"
        )
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
