#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot MolVQE-21 and BenchmarkQC hit iterations versus overlap and gap."""

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOLVQE_CSV = BENCHMARK_ROOT / "plots" / "sector_gap_vs_krylov_iterations.csv"
DEFAULT_BENCHMARK_QC_MANIFEST = Path(
    "/home/mushir.kolathingal/BenchmarkQC/datasets/benchmarkQC/source/manifest.csv"
)
DEFAULT_BENCHMARK_QC_GAP_DIR = (
    BENCHMARK_ROOT / "runs" / "benchmark_qc_51_initial_state_gap"
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
    / "molvqe_benchmarkqc_qrte_qkud_iterations_vs_overlap_gap"
)
EXPECTED_BENCHMARK_QC_GAP_SCHEMA = "benchmark-qc.reference-matched-gap.v1"

FIELDNAMES = [
    "dataset",
    "case_id",
    "method",
    "iteration_to_chemical_accuracy",
    "overlap_weight",
    "negative_log10_overlap",
    "gap_hartree",
    "gap_millihartree",
    "gap_ev",
    "reference_sector",
    "gap_path",
    "summary_path",
]

METHODS = (
    ("QRTE", "qrte_dt0p05_chem_accuracy_iteration", "qrte_hit_chemical_accuracy"),
    ("QKUD", "qkud_eps0p05_chem_accuracy_iteration", "qkud_hit_chemical_accuracy"),
)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path.resolve())


def _positive_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive, got {value!r}")
    return number


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _molvqe_points(path: Path) -> tuple[list[dict[str, Any]], int]:
    points: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        case_id = row["case_id"]
        overlap = _positive_float(row["overlap_weight"], label=f"{case_id} overlap")
        gap_hartree = _positive_float(row["gap_hartree"], label=f"{case_id} gap")
        for method, iteration_key, hit_key in METHODS:
            if str(row.get(hit_key, "")).lower() != "true":
                continue
            iteration = _positive_float(
                row[iteration_key], label=f"{case_id} {method} iteration"
            )
            points.append(
                {
                    "dataset": "MolVQE-21",
                    "case_id": case_id,
                    "method": method,
                    "iteration_to_chemical_accuracy": iteration,
                    "overlap_weight": overlap,
                    "negative_log10_overlap": float(-math.log10(overlap)),
                    "gap_hartree": gap_hartree,
                    "gap_millihartree": float(row["gap_millihartree"]),
                    "gap_ev": _positive_float(row["gap_ev"], label=f"{case_id} gap eV"),
                    "reference_sector": "fixed_n_alpha_n_beta",
                    "gap_path": row.get("gap_path", ""),
                    "summary_path": row.get(
                        "qrte_summary_path" if method == "QRTE" else "qkud_summary_path",
                        "",
                    ),
                }
            )
    return points, len(rows)


def _summary_hit(path: Path) -> tuple[bool, float | None]:
    if not path.exists():
        return False, None
    summary = _load_json(path)
    hit = (
        summary.get("status") == "completed"
        and summary.get("hit_chemical_accuracy") is True
        and summary.get("hit_iteration") is not None
    )
    if not hit:
        return False, None
    return True, _positive_float(summary["hit_iteration"], label=f"{path} hit iteration")


def _benchmark_qc_points(
    manifest_path: Path,
    gap_dir: Path,
    qrte_dir: Path,
    qkud_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    points: list[dict[str, Any]] = []
    missing_hit_gaps: list[str] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    for row in manifest_rows:
        case_id = row["case_id"]
        method_hits: list[tuple[str, float, Path]] = []
        for method, summary_dir in (
            ("QRTE", qrte_dir),
            ("QKUD", qkud_dir),
        ):
            summary_path = summary_dir / case_id / "summary.json"
            hit, iteration = _summary_hit(summary_path)
            if hit and iteration is not None:
                method_hits.append((method, iteration, summary_path))
        if not method_hits:
            continue

        gap_path = gap_dir / case_id / "sector_gap.json"
        if not gap_path.exists():
            missing_hit_gaps.append(f"{case_id}: missing {gap_path}")
            continue
        gap = _load_json(gap_path)
        if gap.get("status") != "completed" or gap.get("schema") != EXPECTED_BENCHMARK_QC_GAP_SCHEMA:
            missing_hit_gaps.append(
                f"{case_id}: expected schema {EXPECTED_BENCHMARK_QC_GAP_SCHEMA}, "
                f"found {gap.get('schema')!r}"
            )
            continue

        reference_difference = abs(
            float(gap["reference_energy_hartree"])
            - float(row["reference_energy_hartree"])
        )
        if reference_difference > 1.0e-7:
            raise ValueError(
                f"{case_id}: gap and manifest references differ by "
                f"{reference_difference:.3e} Ha"
            )
        overlap = _positive_float(gap["overlap_weight"], label=f"{case_id} overlap")
        gap_hartree = _positive_float(gap["gap_hartree"], label=f"{case_id} gap")
        for method, iteration, summary_path in method_hits:
            points.append(
                {
                    "dataset": "BenchmarkQC",
                    "case_id": case_id,
                    "method": method,
                    "iteration_to_chemical_accuracy": iteration,
                    "overlap_weight": overlap,
                    "negative_log10_overlap": float(-math.log10(overlap)),
                    "gap_hartree": gap_hartree,
                    "gap_millihartree": float(gap["gap_millihartree"]),
                    "gap_ev": _positive_float(gap["gap_ev"], label=f"{case_id} gap eV"),
                    "reference_sector": gap["reference_sector"],
                    "gap_path": _relative(gap_path),
                    "summary_path": _relative(summary_path),
                }
            )

    if missing_hit_gaps:
        details = "\n  ".join(missing_hit_gaps)
        raise SystemExit(
            f"Missing or invalid gap data for {len(missing_hit_gaps)} BenchmarkQC hit cases:\n"
            f"  {details}"
        )
    return points, len(manifest_rows)


def _write_csv(points: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for point in points:
            writer.writerow({field: point.get(field, "") for field in FIELDNAMES})
    tmp_path.replace(path)


def _plot(points: list[dict[str, Any]], output_prefix: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "axes.labelsize": 13.0,
            "axes.titlesize": 13.2,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 10.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
        }
    )
    method_colors = {"QRTE": "#D55E00", "QKUD": "#0072B2"}
    dataset_styles = {
        "MolVQE-21": {"marker": "o", "filled": True, "size": 51},
        "BenchmarkQC": {"marker": "D", "filled": False, "size": 58},
    }
    panels = (
        ("overlap_weight", r"Initial overlap, $|\langle\Phi_\mathrm{ref}|\Psi_0\rangle|^2$", "(a) Initial-state overlap"),
        ("gap_ev", r"Reference-sector gap, $E_1-E_0$ (eV)", "(b) Spectral gap"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.35, 4.55), sharey=True)
    paired = {
        (point["dataset"], point["case_id"], point["method"]): point
        for point in points
    }
    cases = sorted({(point["dataset"], point["case_id"]) for point in points})
    for ax, (x_key, xlabel, title) in zip(axes, panels, strict=True):
        for dataset, case_id in cases:
            qrte = paired.get((dataset, case_id, "QRTE"))
            qkud = paired.get((dataset, case_id, "QKUD"))
            if qrte is None or qkud is None:
                continue
            x = float(qrte[x_key])
            ax.plot(
                [x, x],
                [
                    float(qrte["iteration_to_chemical_accuracy"]),
                    float(qkud["iteration_to_chemical_accuracy"]),
                ],
                color="#B6B6B6",
                linewidth=0.6,
                alpha=0.38,
                zorder=1,
            )

        for dataset, dataset_style in dataset_styles.items():
            for method, color in method_colors.items():
                selected = [
                    point
                    for point in points
                    if point["dataset"] == dataset and point["method"] == method
                ]
                if not selected:
                    continue
                facecolor = color if dataset_style["filled"] else "white"
                edgecolor = "white" if dataset_style["filled"] else color
                ax.scatter(
                    [float(point[x_key]) for point in selected],
                    [float(point["iteration_to_chemical_accuracy"]) for point in selected],
                    s=dataset_style["size"],
                    marker=dataset_style["marker"],
                    facecolors=facecolor,
                    edgecolors=edgecolor,
                    linewidths=0.65 if dataset_style["filled"] else 1.35,
                    alpha=0.91,
                    zorder=4 if dataset == "MolVQE-21" else 5,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, which="major", color="#D8D8D8", alpha=0.58, linewidth=0.65)
        ax.grid(True, which="minor", color="#ECECEC", alpha=0.38, linewidth=0.4)
        ax.tick_params(which="both", direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    iterations = [float(point["iteration_to_chemical_accuracy"]) for point in points]
    overlaps = [float(point["overlap_weight"]) for point in points]
    axes[0].set_xlim(min(overlaps) * 0.2, 1.8)
    axes[0].set_ylim(0.8, max(iterations) * 1.22)
    axes[0].set_ylabel("Iterations to chemical accuracy")

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=7.0,
            label=method,
        )
        for method, color in method_colors.items()
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor="#666666" if style["filled"] else "white",
            markeredgecolor="#666666",
            markeredgewidth=1.2,
            markersize=7.0,
            label=dataset,
        )
        for dataset, style in dataset_styles.items()
    ]
    axes[0].legend(
        handles=method_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.93,
    )
    axes[1].legend(
        handles=dataset_handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#D0D0D0",
        framealpha=0.93,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.15, wspace=0.11)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molvqe-csv", type=Path, default=DEFAULT_MOLVQE_CSV)
    parser.add_argument(
        "--benchmark-qc-manifest", type=Path, default=DEFAULT_BENCHMARK_QC_MANIFEST
    )
    parser.add_argument("--benchmark-qc-gap-dir", type=Path, default=DEFAULT_BENCHMARK_QC_GAP_DIR)
    parser.add_argument("--benchmark-qc-qrte-dir", type=Path, default=DEFAULT_BENCHMARK_QC_QRTE_DIR)
    parser.add_argument("--benchmark-qc-qkud-dir", type=Path, default=DEFAULT_BENCHMARK_QC_QKUD_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, molvqe_count = _molvqe_points(args.molvqe_csv)
    benchmark_qc_points, benchmark_qc_count = _benchmark_qc_points(
        args.benchmark_qc_manifest,
        args.benchmark_qc_gap_dir,
        args.benchmark_qc_qrte_dir,
        args.benchmark_qc_qkud_dir,
    )
    points.extend(benchmark_qc_points)
    points.sort(
        key=lambda point: (
            str(point["dataset"]),
            str(point["case_id"]),
            str(point["method"]),
        )
    )
    if molvqe_count != 27 or benchmark_qc_count != 51:
        raise SystemExit(
            f"Expected 27 MolVQE-21 and 51 BenchmarkQC systems, found "
            f"{molvqe_count} and {benchmark_qc_count}"
        )

    csv_path = args.output_prefix.with_suffix(".csv")
    _write_csv(points, csv_path)
    _plot(points, args.output_prefix)

    for dataset in ("MolVQE-21", "BenchmarkQC"):
        counts = {
            method: sum(
                point["dataset"] == dataset and point["method"] == method
                for point in points
            )
            for method in ("QRTE", "QKUD")
        }
        print(f"{dataset}: {counts['QRTE']} QRTE hits and {counts['QKUD']} QKUD hits plotted")
    print("No-hit method/system rows were excluded from the CSV and figure.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
