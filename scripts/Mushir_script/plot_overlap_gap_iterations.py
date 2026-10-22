#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot initial overlap, fixed-sector gap, and Krylov iterations by method."""

from __future__ import annotations

import argparse
import csv
import json
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
import plot_molvqe27_three_krylov_renyi as validated  # noqa: E402


DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "overlap_gap_iterations"
MF_METHOD = "QRTE dt=pi/W_MF"

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
    "summary_path", "history_path", "hamiltonian_cache_path", "hamiltonian_sha256",
    "reference_energy_hartree", "delta_t", "epsilon", "trotter_order", "trotter_steps",
    "descriptor_path", "descriptor_hamiltonian_cache_path", "descriptor_hamiltonian_sha256",
]


def _corrected_co_descriptors(fingerprints):
    """Diagonalize the complete 225-determinant sector of the matching cache."""
    from pyscf import fci, lib
    from scipy.linalg import eigh

    lib.num_threads(1)
    reference_path = validated.CO_ROOT / "reference/reference.json"
    reference = json.loads(reference_path.read_text())
    cache_path = Path(reference["case"]["hamiltonian_cache_path"])
    if validated.sha(cache_path) != reference["case"]["hamiltonian_sha256"]:
        raise ValueError("Corrected CO cache checksum changed")
    with np.load(cache_path, allow_pickle=False) as cache:
        norb, electrons = int(cache["ncas"]), tuple(int(n) for n in cache["nelecas"])
        if (norb, electrons) != (6, (4, 2)):
            raise ValueError("Unexpected corrected CO sector")
        addresses, matrix = fci.direct_spin1.pspace(
            cache["h1ecas"], np.swapaxes(cache["two_mo"], 1, 3), norb, electrons, np=225)
        if not np.array_equal(np.sort(addresses), np.arange(225)):
            raise ValueError("Incomplete corrected CO sector")
        energies, vectors = eigh(matrix)
        energies += float(cache["ecore"])
        if abs(energies[0] - float(cache["reference_energy"])) > 1e-8:
            raise ValueError("Corrected CO ground energy disagrees with reference")
    # The spin-resolved HF determinant occupies the lowest orbitals of each spin.
    alpha = fci.cistring.str2addr(norb, electrons[0], (1 << electrons[0]) - 1)
    beta = fci.cistring.str2addr(norb, electrons[1], (1 << electrons[1]) - 1)
    position = int(np.flatnonzero(addresses == alpha * 15 + beta)[0])
    weight = float(abs(vectors[position, 0]) ** 2)
    probabilities = abs(vectors[:, 0]) ** 2
    gap = float(energies[1] - energies[0])
    descriptor = {
        "case_id": validated.CO, "status": "completed", "solver": "complete-sector PySCF pspace / scipy eigh",
        "hamiltonian_cache_path": validated.relative(cache_path), "hamiltonian_sha256": validated.sha(cache_path),
        "sector_dim": 225, "n_alpha": 4, "n_beta": 2,
        "sector_ground_energy_hartree": float(energies[0]), "first_excited_energy_hartree": float(energies[1]),
        "gap_hartree": gap, "gap_millihartree": gap * 1000, "gap_ev": gap * 27.211386245988,
        "overlap_weight": weight, "h0.25": float(np.log(np.sum(probabilities ** .25)) / .75),
        "reference_energy_hartree": reference["case"]["local_reference_energy_hartree"],
    }
    path = validated.CO_ROOT / "reference/sector_gap.json"
    path.write_text(json.dumps(descriptor, indent=2) + "\n")
    for source in (reference_path, cache_path, path):
        fingerprints[validated.relative(source)] = validated.sha(source)
    return descriptor, path


def _validated_points(rows):
    fingerprints, points, runs = {}, [], []
    corrected_co, corrected_path = _corrected_co_descriptors(fingerprints)
    caches = {r["case_id"]: r for r in gap_data._read_manifest(
        BENCHMARK_ROOT / "hamiltonian_cache/correct_27_systems/manifest.csv")}
    for row in rows:
        cid = row["case_id"]
        descriptor_path = corrected_path if cid == validated.CO else BENCHMARK_ROOT / row["gap_path"]
        descriptor = corrected_co if cid == validated.CO else json.loads(descriptor_path.read_text())
        descriptor_cache = BENCHMARK_ROOT / descriptor["hamiltonian_cache_path"]
        fingerprints[validated.relative(descriptor_path)] = validated.sha(descriptor_path)
        selected = {}
        for series, method, old_key in (("qrte_fixed", "QRTE dt=0.05", "qrte_summary_path"),
                                         ("qkud", "QKUD eps=0.05", "qkud_summary_path"),
                                         ("qrte_mf", MF_METHOD, None)):
            path = (validated.selected_path(cid, series) if cid == validated.CO or series == "qrte_mf"
                    else BENCHMARK_ROOT / row[old_key])
            summary = json.loads(path.read_text())
            if series == "qrte_mf" and cid != validated.CO:
                cached = caches[cid]
            else:
                cache = BENCHMARK_ROOT / summary.get("hamiltonian_cache_path", descriptor["hamiltonian_cache_path"])
                cached = {"case_id": cid, "consolidated_cache_path": validated.relative(cache), "sha256": validated.sha(cache)}
            result = validated.validate_run(path, series, cached, fingerprints)
            if cid == validated.CO and (result["hamiltonian_sha256"] != corrected_co["hamiltonian_sha256"]
                                       or abs(result["reference_energy_hartree"] - corrected_co["reference_energy_hartree"]) > 1e-12):
                raise ValueError("CO result does not match the corrected descriptor/reference")
            selected[series] = result
            runs.append({"case_id": cid, "method": method, **result})
            if result["hit_chemical_accuracy"]:
                points.append({
                    "case_id": cid, "method": method, "iteration_to_chemical_accuracy": result["iteration"],
                    "overlap_weight": descriptor["overlap_weight"], "gap_ev": descriptor["gap_ev"],
                    "gap_millihartree": descriptor["gap_millihartree"],
                    "h0.25": corrected_co["h0.25"] if cid == validated.CO else row["h0.25"],
                    "qkud_only_hit": "false", "source_name": row["source_name"],
                    **{key: result[key] for key in ("summary_path", "hamiltonian_cache_path", "hamiltonian_sha256",
                       "reference_energy_hartree", "delta_t", "epsilon", "trotter_order", "trotter_steps")},
                    "history_path": validated.relative(path.parent / "min_energy_history.csv"),
                    "descriptor_path": validated.relative(descriptor_path),
                    "descriptor_hamiltonian_cache_path": validated.relative(descriptor_cache),
                    "descriptor_hamiltonian_sha256": validated.sha(descriptor_cache),
                })
        for point in points:
            if point["case_id"] == cid and point["method"] == "QKUD eps=0.05":
                point["qkud_only_hit"] = str(not selected["qrte_fixed"]["hit_chemical_accuracy"]).lower()
    points.sort(key=lambda p: (float(p["overlap_weight"]), p["method"], p["case_id"]))
    return points, {"runs": runs, "source_sha256": fingerprints, "corrected_co_descriptors": corrected_co,
                    "chemical_accuracy_hartree": .0016,
                    "source_selection": "Existing fixed-dt QRTE/QKUD sources and coordinates retained except corrected CO; mean-field QRTE uses second-order Trotter, one slice.",
                    "plot_script_sha256": validated.sha(Path(__file__))}


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
    if method == MF_METHOD:
        return r"QRTE $\delta t=\pi/W_{\mathrm{MF}}$"
    if method.startswith("QRTE"):
        return r"QRTE $\delta t=0.05$ a.u."
    return r"QKUD $\epsilon=0.05$ a.u."


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
            "font.size": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.95,
            "axes.labelsize": 13.2,
            "axes.titlesize": 13.2,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "legend.fontsize": 10.5,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "xtick.major.size": 4.4,
            "ytick.major.size": 4.4,
            "xtick.minor.size": 2.4,
            "ytick.minor.size": 2.4,
        }
    )

    fig, ax = plt.subplots(figsize=(6.7, 5.05), constrained_layout=True)
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
            fontsize=9.0,
            color="#333333",
            ha="left",
            va="bottom",
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Initial overlap, $|\langle\Phi_\mathrm{ref}|\Psi_0\rangle|^2$")
    ax.set_ylabel(r"Spectral gap, $E_1 - E_0$ (eV)")
    ax.set_title(f"{_method_label(method)}")
    ax.grid(True, which="major", color="#D8D8D8", alpha=0.55, linewidth=0.7)
    ax.grid(True, which="minor", color="#E8E8E8", alpha=0.35, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.015)
    colorbar.set_label(r"$N_{\mathrm{QKrylov}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)")
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
    _ = iteration_norm
    _plot_iteration_drivers(points, png_path, pdf_path)


def _plot_iteration_drivers(
    points: list[dict[str, Any]],
    png_path: Path,
    pdf_path: Path,
) -> None:
    methods = ("QRTE dt=0.05", "QKUD eps=0.05", MF_METHOD)
    by_case_method = {(point["case_id"], point["method"]): point for point in points}
    cases = sorted({point["case_id"] for point in points})
    qkud_only_points = [point for point in points if point["qkud_only_hit"] == "true"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.95,
            "axes.labelsize": 12.8,
            "xtick.labelsize": 11.2,
            "ytick.labelsize": 11.2,
            "legend.fontsize": 10.6,
            "xtick.major.width": 0.95,
            "ytick.major.width": 0.95,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "xtick.major.size": 4.6,
            "ytick.major.size": 4.6,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.15, 4.15), sharey=True, constrained_layout=True)
    panels = (
        (
            axes[0],
            "overlap_weight",
            r"Initial overlap, $|\langle\Phi_\mathrm{ref}|\Psi_0\rangle|^2$",
            "(a) Initial overlap",
            "linear",
        ),
        (
            axes[1],
            "gap_ev",
            r"Spectral gap, $E_1 - E_0$ (eV)",
            "(b) Spectral gap",
            "log",
        ),
    )

    method_style = {
        "QRTE dt=0.05": {
            "marker": "s",
            "facecolors": "none",
            "edgecolors": "#D55E00",
            "linewidths": 1.25,
            "s": 58,
            "zorder": 5,
        },
        "QKUD eps=0.05": {
            "marker": "^",
            "c": "#0072B2",
            "edgecolor": "#333333",
            "linewidth": 0.35,
            "s": 54,
            "zorder": 6,
        },
        MF_METHOD: {
            "marker": "D", "facecolors": "none", "edgecolors": "#009E73",
            "linewidths": 1.1, "s": 46, "zorder": 7,
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
            if method == "QKUD eps=0.05":
                method_points = [point for point in method_points if point["qkud_only_hit"] != "true"]
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
                s=145,
                marker="*",
                facecolors="#F0E442",
                edgecolors="#2F2F2F",
                linewidth=0.85,
                zorder=8,
            )

        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        if xscale == "log":
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(0.75, 90.0)
        ax.grid(True, which="major", color="#D6D6D6", alpha=0.48, linewidth=0.6)
        ax.grid(True, which="minor", color="#ECECEC", alpha=0.32, linewidth=0.38)
        ax.tick_params(which="both", direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_xlim(0.43, 1.005)
    axes[1].set_xlim(0.09, 13.0)
    axes[0].set_ylabel(r"$N_{\mathrm{QKrylov}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)")

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=1.25, markersize=6.4),
        Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="#0072B2", markeredgecolor="#333333", markeredgewidth=0.35, markersize=6.4),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="none", markeredgecolor="#009E73", markeredgewidth=1.1, markersize=6.0),
    ]
    legend_labels = [
        r"QRTE $\delta t=0.05$ a.u.",
        r"QKUD $\epsilon=0.05$ a.u.",
        r"QRTE $\delta t=\pi/W_{\mathrm{MF}}$",
    ]
    legend = axes[0].legend(
        legend_handles,
        legend_labels,
        frameon=True,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(0.025, 0.04),
        handlelength=1.25,
        handletextpad=0.55,
        labelspacing=0.42,
        borderpad=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor("#D1D5DB")
    legend.get_frame().set_linewidth(0.65)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
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
    points, audit = _validated_points(rows)
    for path in (args.manifest, BENCHMARK_ROOT / "hamiltonian_cache/correct_27_systems/manifest.csv"):
        audit["source_sha256"][validated.relative(path)] = validated.sha(path)
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
    args.output_prefix.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
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
    print(f"Mean-field timestep QRTE points: {sum(point['method'] == MF_METHOD for point in points)}")
    print(f"Highlighted QKUD-only hits: {', '.join(qkud_only) if qkud_only else 'none'}")
    if missing:
        print(f"Missing or invalid sector-gap rows skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()
