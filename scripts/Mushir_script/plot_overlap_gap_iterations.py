#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot initial overlap, fixed-sector gap, and Krylov iterations by method."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import plot_sector_gap_vs_iterations as gap_data  # noqa: E402


DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "overlap_gap_iterations"

POINT_FIELDNAMES = [
    "case_id",
    "method",
    "iteration_to_chemical_accuracy",
    "overlap_weight",
    "gap_ev",
    "gap_millihartree",
    "h0.25",
    "qkud_only_hit",
    "source_name",
]


def _positive_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number > 0.0 and math.isfinite(number):
        return number
    return None


def _build_point_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        for method, iteration_key, hit_key in (
            ("QRTE dt=0.05", "qrte_dt0p05_chem_accuracy_iteration", "qrte_hit_chemical_accuracy"),
            ("QKUD eps=0.05", "qkud_eps0p05_chem_accuracy_iteration", "qkud_hit_chemical_accuracy"),
        ):
            if row[hit_key] != "true":
                continue
            iteration = _positive_number(row[iteration_key])
            if iteration is None:
                continue
            points.append(
                {
                    "case_id": row["case_id"],
                    "method": method,
                    "iteration_to_chemical_accuracy": iteration,
                    "overlap_weight": float(row["overlap_weight"]),
                    "gap_ev": float(row["gap_ev"]),
                    "gap_millihartree": float(row["gap_millihartree"]),
                    "h0.25": float(row["h0.25"]),
                    "qkud_only_hit": row["qkud_only_hit"] if method.startswith("QKUD") else "false",
                    "source_name": row["source_name"],
                }
            )
    points.sort(key=lambda point: (float(point["overlap_weight"]), point["method"], point["case_id"]))
    return points


def _write_points(points: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)


def _points_for_method(points: list[dict[str, Any]], method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    kept = [point for point in points if point["method"] == method]
    if not kept:
        return np.asarray([]), np.asarray([]), np.asarray([]), []
    x = np.asarray([float(point["overlap_weight"]) for point in kept], dtype=float)
    y = np.asarray([float(point["gap_ev"]) for point in kept], dtype=float)
    color = np.asarray([float(point["iteration_to_chemical_accuracy"]) for point in kept], dtype=float)
    return x, y, color, kept


def _method_label(method: str) -> str:
    if method.startswith("QRTE"):
        return r"QRTE $\Delta t=0.05$"
    return r"QKUD $\epsilon=0.05$"


def _suffixed_path(prefix: Path, slug: str, suffix: str) -> Path:
    return prefix.parent / f"{prefix.name}_{slug}{suffix}"


def _display_case_label(case_id: str) -> str:
    labels = {
        "N2_equi_cc_pVTZ_2": r"$\mathrm{N_2}$",
        "fenoco3_minus": r"$\mathrm{[FeNO(CO)_3]^-}$",
        "feno_2plus": r"$\mathrm{[FeNO]^{2+}}$",
        "FeO": r"$\mathrm{FeO}$",
    }
    return labels.get(case_id, case_id)


def _driver_annotation_offset(case_id: str, x_key: str) -> tuple[int, int]:
    offsets = {
        ("feno_2plus", "gap_ev"): (-6, 6),
        ("feno_2plus", "overlap_weight"): (7, 6),
        ("FeO", "gap_ev"): (7, 7),
        ("FeO", "overlap_weight"): (-6, 5),
        ("fenoco3_minus", "gap_ev"): (6, 5),
        ("fenoco3_minus", "overlap_weight"): (6, 5),
        ("N2_equi_cc_pVTZ_2", "gap_ev"): (6, 5),
        ("N2_equi_cc_pVTZ_2", "overlap_weight"): (6, 5),
    }
    return offsets.get((case_id, x_key), (4, 4))


def _driver_annotation_ha(case_id: str, x_key: str) -> str:
    if (case_id, x_key) in {
        ("feno_2plus", "gap_ev"),
        ("FeO", "overlap_weight"),
    }:
        return "right"
    return "left"


def _plot_method(
    points: list[dict[str, Any]],
    method: str,
    png_path: Path,
    pdf_path: Path,
    *,
    iteration_norm: LogNorm,
) -> None:
    x_values, y_values, iterations, method_points = _points_for_method(points, method)
    if not method_points:
        raise SystemExit(f"No valid points found for {method}.")

    cmap = "plasma"

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.85,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.85), constrained_layout=True)
    scatter = ax.scatter(
        x_values,
        y_values,
        c=iterations,
        cmap=cmap,
        norm=iteration_norm,
        s=68,
        marker="s" if method.startswith("QRTE") else "^",
        edgecolor="#333333",
        linewidth=0.4,
        zorder=4,
    )

    qkud_only_points = [point for point in method_points if point["qkud_only_hit"] == "true"]
    if method.startswith("QKUD") and qkud_only_points:
        ax.scatter(
            [float(point["overlap_weight"]) for point in qkud_only_points],
            [float(point["gap_ev"]) for point in qkud_only_points],
            c=[float(point["iteration_to_chemical_accuracy"]) for point in qkud_only_points],
            cmap=cmap,
            norm=iteration_norm,
            s=185,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=7,
        )

    for point in method_points:
        iteration = float(point["iteration_to_chemical_accuracy"])
        should_label = iteration >= 50.0 or point["qkud_only_hit"] == "true"
        if not should_label:
            continue
        ax.annotate(
            point["case_id"],
            (float(point["overlap_weight"]), float(point["gap_ev"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.8,
            color="#333333",
            ha="left",
            va="bottom",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$|\langle\Phi_\mathrm{ref}|\Psi_0\rangle|^2$")
    ax.set_ylabel(r"Fixed-sector gap, $E_1 - E_0$ (eV)")
    ax.set_title(f"{_method_label(method)}")
    ax.grid(True, which="major", color="#D8D8D8", alpha=0.55, linewidth=0.7)
    ax.grid(True, which="minor", color="#E8E8E8", alpha=0.35, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label("Iterations to chemical accuracy")
    colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())

    if method.startswith("QKUD") and qkud_only_points:
        legend_handles = [
            Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#BDBDBD", markeredgecolor="black", markersize=12.0),
        ]
        ax.legend(legend_handles, ["QKUD hit; QRTE not hit"], frameon=False, loc="best", handlelength=1.3)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_combined(
    points: list[dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
    *,
    iteration_norm: LogNorm,
) -> None:
    qrte_x, qrte_y, qrte_iterations, qrte_points = _points_for_method(points, "QRTE dt=0.05")
    qkud_x, qkud_y, qkud_iterations, qkud_points = _points_for_method(points, "QKUD eps=0.05")
    if not qrte_points and not qkud_points:
        raise SystemExit("No valid QRTE or QKUD points found.")

    cmap = "plasma"
    cmap_obj = plt.get_cmap(cmap)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.85,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.major.width": 0.85,
            "ytick.major.width": 0.85,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
        }
    )

    fig, ax = plt.subplots(figsize=(6.9, 5.0), constrained_layout=True)
    if qrte_points:
        ax.scatter(
            qrte_x,
            qrte_y,
            facecolors="none",
            edgecolors=cmap_obj(iteration_norm(qrte_iterations)),
            s=105,
            marker="s",
            linewidth=1.35,
            zorder=5,
        )

    qkud_scatter = None
    if qkud_points:
        qkud_scatter = ax.scatter(
            qkud_x,
            qkud_y,
            c=qkud_iterations,
            cmap=cmap,
            norm=iteration_norm,
            s=68,
            marker="^",
            edgecolor="#333333",
            linewidth=0.35,
            zorder=4,
        )

    qkud_only_points = [point for point in qkud_points if point["qkud_only_hit"] == "true"]
    if qkud_only_points:
        ax.scatter(
            [float(point["overlap_weight"]) for point in qkud_only_points],
            [float(point["gap_ev"]) for point in qkud_only_points],
            c=[float(point["iteration_to_chemical_accuracy"]) for point in qkud_only_points],
            cmap=cmap,
            norm=iteration_norm,
            s=185,
            marker="*",
            edgecolor="black",
            linewidth=0.8,
            zorder=7,
        )

    labeled: set[str] = set()
    for point in points:
        iteration = float(point["iteration_to_chemical_accuracy"])
        should_label = iteration >= 50.0 or point["qkud_only_hit"] == "true"
        if not should_label or point["case_id"] in labeled:
            continue
        labeled.add(point["case_id"])
        ax.annotate(
            point["case_id"],
            (float(point["overlap_weight"]), float(point["gap_ev"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.8,
            color="#333333",
            ha="left",
            va="bottom",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$|\langle\Phi_\mathrm{ref}|\Psi_0\rangle|^2$")
    ax.set_ylabel(r"Fixed-sector gap, $E_1 - E_0$ (eV)")
    ax.set_title(r"QRTE $\Delta t=0.05$ and QKUD $\epsilon=0.05$")
    ax.grid(True, which="major", color="#D8D8D8", alpha=0.55, linewidth=0.7)
    ax.grid(True, which="minor", color="#E8E8E8", alpha=0.35, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if qkud_scatter is not None:
        colorbar = fig.colorbar(qkud_scatter, ax=ax, pad=0.015)
        colorbar.set_label("Iterations to chemical accuracy")
        colorbar.ax.yaxis.set_major_formatter(ScalarFormatter())

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="#777777", markersize=8.5),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#777777", markeredgecolor="#333333", markersize=8.0),
    ]
    legend_labels = [
        rf"QRTE $\Delta t=0.05$ (n={len(qrte_points)})",
        rf"QKUD $\epsilon=0.05$ (n={len(qkud_points)})",
    ]
    if qkud_only_points:
        legend_handles.append(
            Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#BDBDBD", markeredgecolor="black", markersize=12.0)
        )
        legend_labels.append("QKUD hit; QRTE not hit")
    ax.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        handlelength=1.3,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _plot_iteration_drivers(
    points: list[dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    methods = ("QRTE dt=0.05", "QKUD eps=0.05")
    by_case_method = {(point["case_id"], point["method"]): point for point in points}
    cases = sorted({point["case_id"] for point in points})
    qkud_only_points = [point for point in points if point["qkud_only_hit"] == "true"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.75,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
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

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.65), sharey=True, constrained_layout=True)
    panels = (
        (
            axes[0],
            "overlap_weight",
            r"$|\langle\Phi_\mathrm{ref}|\Psi_0\rangle|^2$",
            "(a) Initial overlap",
            "linear",
        ),
        (
            axes[1],
            "gap_ev",
            r"Fixed-sector gap, $E_1 - E_0$ (eV)",
            "(b) Spectral gap",
            "log",
        ),
    )

    method_style = {
        "QRTE dt=0.05": {
            "marker": "s",
            "facecolors": "none",
            "edgecolors": "#D55E00",
            "linewidths": 1.05,
            "s": 46,
            "zorder": 5,
        },
        "QKUD eps=0.05": {
            "marker": "^",
            "c": "#0072B2",
            "edgecolor": "#333333",
            "linewidth": 0.25,
            "s": 42,
            "zorder": 6,
        },
    }

    for ax, x_key, xlabel, title, xscale in panels:
        for case_id in cases:
            qrte = by_case_method.get((case_id, "QRTE dt=0.05"))
            qkud = by_case_method.get((case_id, "QKUD eps=0.05"))
            if qrte is None or qkud is None:
                continue
            x_value = float(qrte[x_key])
            ax.plot(
                [x_value, x_value],
                [
                    float(qkud["iteration_to_chemical_accuracy"]),
                    float(qrte["iteration_to_chemical_accuracy"]),
                ],
                color="#B8B8B8",
                linewidth=0.65,
                alpha=0.45,
                zorder=2,
            )

        for method in methods:
            method_points = [point for point in points if point["method"] == method]
            if not method_points:
                continue
            ax.scatter(
                [float(point[x_key]) for point in method_points],
                [float(point["iteration_to_chemical_accuracy"]) for point in method_points],
                **method_style[method],
            )

        if qkud_only_points:
            ax.scatter(
                [float(point[x_key]) for point in qkud_only_points],
                [float(point["iteration_to_chemical_accuracy"]) for point in qkud_only_points],
                s=120,
                marker="*",
                color="#F0E442",
                edgecolor="black",
                linewidth=0.8,
                zorder=8,
            )

        for point in points:
            iteration = float(point["iteration_to_chemical_accuracy"])
            should_label = iteration >= 50.0 or point["qkud_only_hit"] == "true"
            if not should_label:
                continue
            ax.annotate(
                _display_case_label(str(point["case_id"])),
                (float(point[x_key]), iteration),
                xytext=_driver_annotation_offset(str(point["case_id"]), str(x_key)),
                textcoords="offset points",
                fontsize=7.2,
                color="#333333",
                ha=_driver_annotation_ha(str(point["case_id"]), str(x_key)),
                va="bottom",
            )

        ax.set_title(title, loc="left", pad=5)
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        if xscale == "log":
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(0.75, 90.0)
        ax.grid(True, which="major", color="#D6D6D6", alpha=0.46, linewidth=0.55)
        ax.grid(True, which="minor", color="#ECECEC", alpha=0.30, linewidth=0.35)
        ax.tick_params(which="both", direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_xlim(0.43, 1.005)
    axes[1].set_xlim(0.09, 13.0)
    axes[0].set_ylabel("Iterations to chemical accuracy")

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="#D55E00", markersize=5.7),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#0072B2", markeredgecolor="#333333", markersize=5.7),
        Line2D([0], [0], color="#B8B8B8", linewidth=0.9),
    ]
    legend_labels = [
        rf"QRTE $\Delta t=0.05$",
        rf"QKUD $\epsilon=0.05$",
        "same molecule",
    ]
    if qkud_only_points:
        legend_handles.append(
            Line2D([0], [0], marker="*", linestyle="none", markerfacecolor="#F0E442", markeredgecolor="black", markersize=9.0)
        )
        legend_labels.append("QKUD hit; QRTE not hit")
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.155),
        handlelength=1.4,
        columnspacing=1.35,
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=gap_data.DEFAULT_MANIFEST)
    parser.add_argument("--gap-dir", type=Path, default=gap_data.DEFAULT_GAP_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--expected-target-count", type=int, default=27)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = gap_data._read_manifest(args.manifest)
    target_rows = gap_data._target_manifest_rows(manifest_rows)
    if len(target_rows) != int(args.expected_target_count):
        raise SystemExit(f"Expected {args.expected_target_count} target manifest rows, found {len(target_rows)}")

    rows, missing = gap_data._build_rows(manifest_rows, args.gap_dir)
    if missing and not args.allow_missing:
        missing_text = "\n  ".join(missing)
        raise SystemExit(
            f"Missing or invalid sector gap JSON for {len(missing)} target systems:\n  {missing_text}\n"
            "Rerun with --allow-missing for a preview plot."
        )
    points = _build_point_rows(rows)
    if not points:
        raise SystemExit("No valid method points found.")

    csv_path = args.output_prefix.with_suffix(".csv")
    iteration_values = np.asarray([float(point["iteration_to_chemical_accuracy"]) for point in points], dtype=float)
    iteration_min = max(1.0, float(iteration_values.min()))
    iteration_max = float(iteration_values.max())
    if iteration_max <= iteration_min:
        iteration_max = iteration_min + 1.0
    iteration_norm = LogNorm(vmin=iteration_min, vmax=iteration_max)

    combined_png_path = args.output_prefix.with_suffix(".png")
    combined_pdf_path = args.output_prefix.with_suffix(".pdf")
    drivers_png_path = _suffixed_path(args.output_prefix, "drivers", ".png")
    drivers_pdf_path = _suffixed_path(args.output_prefix, "drivers", ".pdf")
    method_paths = {
        "QRTE dt=0.05": (
            _suffixed_path(args.output_prefix, "qrte", ".png"),
            _suffixed_path(args.output_prefix, "qrte", ".pdf"),
        ),
        "QKUD eps=0.05": (
            _suffixed_path(args.output_prefix, "qkud", ".png"),
            _suffixed_path(args.output_prefix, "qkud", ".pdf"),
        ),
    }
    _write_points(points, csv_path)
    _plot_combined(points, combined_png_path, combined_pdf_path, iteration_norm=iteration_norm)
    _plot_iteration_drivers(points, drivers_png_path, drivers_pdf_path)
    for method, (png_path, pdf_path) in method_paths.items():
        _plot_method(points, method, png_path, pdf_path, iteration_norm=iteration_norm)

    qkud_count = sum(1 for point in points if point["method"] == "QKUD eps=0.05")
    qrte_count = sum(1 for point in points if point["method"] == "QRTE dt=0.05")
    qkud_only = [point["case_id"] for point in points if point["qkud_only_hit"] == "true"]
    print(f"Wrote {csv_path}")
    print(f"Wrote {combined_png_path}")
    print(f"Wrote {combined_pdf_path}")
    print(f"Wrote {drivers_png_path}")
    print(f"Wrote {drivers_pdf_path}")
    for png_path, pdf_path in method_paths.values():
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")
    print(f"Included {len(rows)} systems and {len(points)} method points.")
    print(f"QKUD points: {qkud_count}")
    print(f"QRTE points: {qrte_count}")
    print(f"Highlighted QKUD-only hits: {', '.join(qkud_only) if qkud_only else 'none'}")
    if missing:
        print(f"Missing or invalid sector-gap rows skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()
