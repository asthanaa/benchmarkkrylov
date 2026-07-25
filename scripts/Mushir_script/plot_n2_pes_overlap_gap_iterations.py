#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot N2 PES iterations to chemical accuracy versus overlap and gap."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(TMPDIR / "benchmarkkrylov-pycache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QRTE_DIR = BENCHMARK_ROOT / "runs" / "n2_pes_qrte_natorb_dt0.05"
DEFAULT_QKUD_DIR = BENCHMARK_ROOT / "runs" / "n2_pes_qkud_natorb_eps0.05"
DEFAULT_GAP_DIR = BENCHMARK_ROOT / "runs" / "n2_pes_initial_state_gap"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "n2_pes_overlap_gap_iterations"
DEFAULT_EQUILIBRIUM_QRTE_DIR = BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.05"
DEFAULT_EQUILIBRIUM_QRTE_RERUN_DIR = BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.05_100iter"
DEFAULT_EQUILIBRIUM_QKUD_DIR = BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.05"
DEFAULT_EQUILIBRIUM_GAP_DIR = BENCHMARK_ROOT / "runs" / "initial_state_gap"
EQUILIBRIUM_BOND_LENGTH_ANGSTROM = 1.20
EQUILIBRIUM_CASE_IDS = ("N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2")
CASE_RE = re.compile(r"^N2_R(?P<bond>[0-9]+p[0-9]+)_cc_pVTZ_(?P<spin>[02])$")
EQUILIBRIUM_CASE_RE = re.compile(r"^N2_equi_cc_pVTZ_(?P<spin>[02])$")

FIELDNAMES = [
    "case_id",
    "bond_length_angstrom",
    "spin",
    "point_kind",
    "method",
    "hit_chemical_accuracy",
    "iteration_to_chemical_accuracy",
    "steps_completed",
    "overlap_weight",
    "negative_log10_overlap",
    "gap_hartree",
    "gap_millihartree",
    "gap_ev",
    "gap_path",
    "summary_path",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BENCHMARK_ROOT))
    except ValueError:
        return str(path)


def _case_metadata(case_id: str) -> tuple[float, int]:
    match = CASE_RE.match(case_id)
    if match is not None:
        bond = float(match.group("bond").replace("p", "."))
        spin = int(match.group("spin"))
        return bond, spin
    equilibrium_match = EQUILIBRIUM_CASE_RE.match(case_id)
    if equilibrium_match is not None:
        return EQUILIBRIUM_BOND_LENGTH_ANGSTROM, int(equilibrium_match.group("spin"))
    raise ValueError(f"Unexpected N2 PES/equilibrium case id: {case_id}")


def _case_kind(case_id: str) -> str:
    return "equilibrium" if EQUILIBRIUM_CASE_RE.match(case_id) else "pes"


def _summary_hit(summary: dict[str, Any] | None) -> bool:
    return bool(
        summary is not None
        and summary.get("status") == "completed"
        and summary.get("hit_chemical_accuracy") is True
        and summary.get("hit_iteration") is not None
    )


def _float_or_blank(value: Any) -> float | str:
    if value in (None, ""):
        return ""
    number = float(value)
    return number if math.isfinite(number) else ""


def _discover_case_ids(gap_dir: Path) -> list[str]:
    case_ids = [path.parent.name for path in gap_dir.glob("*/sector_gap.json") if CASE_RE.match(path.parent.name)]
    return sorted(case_ids, key=lambda case_id: (*_case_metadata(case_id), case_id))


def _preferred_summary(
    case_id: str,
    summary_dir: Path,
    *,
    rerun_dir: Path | None = None,
) -> tuple[Path, dict[str, Any] | None]:
    base_path = summary_dir / case_id / "summary.json"
    base_summary = _load_json(base_path) if base_path.exists() else None

    rerun_path = None if rerun_dir is None else rerun_dir / case_id / "summary.json"
    rerun_summary = _load_json(rerun_path) if rerun_path is not None and rerun_path.exists() else None
    if _summary_hit(rerun_summary):
        return rerun_path, rerun_summary
    if base_summary is not None:
        return base_path, base_summary
    if rerun_path is not None:
        return rerun_path, rerun_summary
    return base_path, None


def _append_case_rows(
    rows: list[dict[str, Any]],
    missing: list[str],
    *,
    case_id: str,
    gap_dir: Path,
    qrte_dir: Path,
    qkud_dir: Path,
    qrte_rerun_dir: Path | None = None,
) -> None:
    bond, spin = _case_metadata(case_id)
    point_kind = _case_kind(case_id)
    gap_path = gap_dir / case_id / "sector_gap.json"
    if not gap_path.exists():
        missing.append(f"{case_id}: gap")
        return
    gap = _load_json(gap_path)
    if gap.get("status") != "completed" or float(gap.get("gap_hartree", 0.0)) <= 0.0:
        missing.append(f"{case_id}: invalid gap")
        return

    for method, summary_dir, method_label in (
        ("qrte", qrte_dir, "QRTE dt=0.05"),
        ("qkud", qkud_dir, "QKUD eps=0.05"),
    ):
        summary_path, summary = _preferred_summary(
            case_id,
            summary_dir,
            rerun_dir=qrte_rerun_dir if method == "qrte" else None,
        )
        hit = _summary_hit(summary)
        if summary is None:
            missing.append(f"{case_id}: {method} summary")
        rows.append(
            {
                "case_id": case_id,
                "bond_length_angstrom": float(bond),
                "spin": int(spin),
                "point_kind": point_kind,
                "method": method_label,
                "method_key": method,
                "hit_chemical_accuracy": str(hit).lower(),
                "iteration_to_chemical_accuracy": _float_or_blank(summary.get("hit_iteration") if hit else ""),
                "steps_completed": _float_or_blank(None if summary is None else summary.get("n_steps_completed")),
                "overlap_weight": float(gap["overlap_weight"]),
                "negative_log10_overlap": float(gap["negative_log10_overlap"]),
                "gap_hartree": float(gap["gap_hartree"]),
                "gap_millihartree": float(gap["gap_millihartree"]),
                "gap_ev": float(gap["gap_ev"]),
                "gap_path": _relative(gap_path),
                "summary_path": _relative(summary_path) if summary_path.exists() else "",
            }
        )


def _build_rows(
    *,
    qrte_dir: Path,
    qkud_dir: Path,
    gap_dir: Path,
    equilibrium_qrte_dir: Path,
    equilibrium_qkud_dir: Path,
    equilibrium_gap_dir: Path,
    equilibrium_qrte_rerun_dir: Path | None,
    include_equilibrium: bool,
    expected_count: int,
    allow_missing: bool,
) -> list[dict[str, Any]]:
    case_ids = _discover_case_ids(gap_dir)
    if len(case_ids) != expected_count and not allow_missing:
        raise SystemExit(f"Expected {expected_count} N2 PES gap cases, found {len(case_ids)} in {gap_dir}")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in case_ids:
        _append_case_rows(
            rows,
            missing,
            case_id=case_id,
            gap_dir=gap_dir,
            qrte_dir=qrte_dir,
            qkud_dir=qkud_dir,
        )

    if include_equilibrium:
        for case_id in EQUILIBRIUM_CASE_IDS:
            _append_case_rows(
                rows,
                missing,
                case_id=case_id,
                gap_dir=equilibrium_gap_dir,
                qrte_dir=equilibrium_qrte_dir,
                qkud_dir=equilibrium_qkud_dir,
                qrte_rerun_dir=equilibrium_qrte_rerun_dir,
            )

    if missing and not allow_missing:
        missing_text = "\n  ".join(missing)
        raise SystemExit(f"Missing or invalid N2 PES inputs:\n  {missing_text}\nUse --allow-missing for a preview.")

    rows.sort(key=lambda row: (row["spin"], row["bond_length_angstrom"], row["point_kind"], row["method"]))
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["hit_chemical_accuracy"] == "true" and row["iteration_to_chemical_accuracy"] not in ("", None)
    ]


def _plot_y(row: dict[str, Any]) -> float | None:
    if row["hit_chemical_accuracy"] == "true" and row["iteration_to_chemical_accuracy"] not in ("", None):
        return float(row["iteration_to_chemical_accuracy"])
    if row["steps_completed"] not in ("", None):
        return float(row["steps_completed"])
    return None


def _display_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _plot_y(row) is not None]


def _marker_for(row: dict[str, Any]) -> str:
    if row["method_key"] == "qrte":
        return "o" if int(row["spin"]) == 0 else "s"
    return "s" if int(row["spin"]) == 0 else "o"


def _method_spin_label(method_key: str, spin: int) -> str:
    method = "QRTE" if method_key == "qrte" else "QKUD"
    return rf"{method}, spin {spin}"


def _plot(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    points = _hit_rows(rows)
    if not points:
        raise SystemExit("No N2 PES methods hit chemical accuracy; no iteration plot can be made.")
    display_rows = _display_rows(rows)
    miss_rows = [row for row in display_rows if row["hit_chemical_accuracy"] != "true"]

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 450,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )

    bond_values = [float(row["bond_length_angstrom"]) for row in rows]
    unique_bonds = sorted(set(bond_values))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(unique_bonds), vmax=max(unique_bonds))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.85), sharey=True, constrained_layout=True)
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

    by_case_method = {(row["case_id"], row["method_key"]): row for row in display_rows}
    for ax, x_key, xlabel, title, xscale in panels:
        for case_id in sorted({row["case_id"] for row in display_rows}):
            qrte = by_case_method.get((case_id, "qrte"))
            qkud = by_case_method.get((case_id, "qkud"))
            if qrte is None or qkud is None:
                continue
            x_value = float(qrte[x_key])
            qrte_y = _plot_y(qrte)
            qkud_y = _plot_y(qkud)
            if qrte_y is None or qkud_y is None:
                continue
            ax.plot(
                [x_value, x_value],
                [qrte_y, qkud_y],
                color="#A8A8A8",
                linewidth=0.65,
                alpha=0.42,
                zorder=1,
            )

        for method_key in ("qrte", "qkud"):
            for spin in (0, 2):
                subset = [
                    row
                    for row in points
                    if row["method_key"] == method_key and int(row["spin"]) == spin
                ]
                if not subset:
                    continue
                colors = cmap(norm([float(row["bond_length_angstrom"]) for row in subset]))
                if method_key == "qrte":
                    ax.scatter(
                        [float(row[x_key]) for row in subset],
                        [float(row["iteration_to_chemical_accuracy"]) for row in subset],
                        marker=_marker_for(subset[0]),
                        facecolors="none",
                        edgecolors=colors,
                        linewidths=1.35,
                        s=62,
                        zorder=4,
                    )
                else:
                    ax.scatter(
                        [float(row[x_key]) for row in subset],
                        [float(row["iteration_to_chemical_accuracy"]) for row in subset],
                        marker=_marker_for(subset[0]),
                        c=colors,
                        edgecolors="#303030",
                        linewidths=0.35,
                        s=58,
                        zorder=5,
                    )

        if miss_rows:
            panel_misses = [row for row in miss_rows if _plot_y(row) is not None]
            ax.scatter(
                [float(row[x_key]) for row in panel_misses],
                [float(_plot_y(row)) for row in panel_misses],
                marker="X",
                c=cmap(norm([float(row["bond_length_angstrom"]) for row in panel_misses])),
                edgecolors="black",
                linewidths=0.55,
                s=82,
                zorder=7,
            )

        ax.set_title(title, loc="left", pad=5)
        ax.set_xlabel(xlabel)
        ax.set_xscale(xscale)
        ax.set_yscale("log")
        if xscale == "log":
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(True, which="major", color="#D8D8D8", alpha=0.52, linewidth=0.6)
        ax.grid(True, which="minor", color="#ECECEC", alpha=0.35, linewidth=0.4)
        ax.tick_params(which="both", direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    y_values = np.asarray([float(_plot_y(row)) for row in display_rows if _plot_y(row) is not None], dtype=float)
    ymin = max(0.75, float(y_values.min()) * 0.82)
    ymax = float(y_values.max()) * 1.35
    if ymax <= ymin:
        ymax = ymin + 1.0
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_ylabel("Iterations to chemical accuracy")

    overlap_values = np.asarray([float(row["overlap_weight"]) for row in points], dtype=float)
    overlap_pad = max(0.005, 0.04 * float(overlap_values.max() - overlap_values.min()))
    axes[0].set_xlim(max(0.0, float(overlap_values.min()) - overlap_pad), min(1.0, float(overlap_values.max()) + overlap_pad))

    gap_values = np.asarray([float(row["gap_ev"]) for row in points], dtype=float)
    axes[1].set_xlim(max(0.001, float(gap_values.min()) * 0.82), float(gap_values.max()) * 1.22)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#4B5563", markersize=6.2),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="none", markeredgecolor="#4B5563", markersize=6.0),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor="#4B5563", markeredgecolor="#303030", markersize=6.0),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#4B5563", markeredgecolor="#303030", markersize=6.0),
    ]
    legend_labels = [
        _method_spin_label("qrte", 0),
        _method_spin_label("qrte", 2),
        _method_spin_label("qkud", 0),
        _method_spin_label("qkud", 2),
    ]
    if miss_rows:
        legend_handles.append(
            Line2D([0], [0], marker="X", linestyle="none", markerfacecolor="#4B5563", markeredgecolor="black", markersize=6.8)
        )
        legend_labels.append("no hit; final step")
    for ax in axes:
        legend = ax.legend(
            legend_handles,
            legend_labels,
            frameon=True,
            loc="upper right",
            bbox_to_anchor=(1.02, 0.985),
            borderaxespad=0.0,
            handlelength=1.2,
            handletextpad=0.5,
            labelspacing=0.32,
            borderpad=0.38,
            fontsize=8.2,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.88)
        legend.get_frame().set_edgecolor("#D1D5DB")
        legend.get_frame().set_linewidth(0.65)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=axes, shrink=0.92, pad=0.018, aspect=28)
    colorbar.set_label(r"N-N bond distance, $R$ ($\AA$)")
    colorbar.set_ticks(unique_bonds)
    colorbar.set_ticklabels([f"{value:.2f}" for value in unique_bonds])
    colorbar.outline.set_linewidth(0.65)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrte-dir", type=Path, default=DEFAULT_QRTE_DIR)
    parser.add_argument("--qkud-dir", type=Path, default=DEFAULT_QKUD_DIR)
    parser.add_argument("--gap-dir", type=Path, default=DEFAULT_GAP_DIR)
    parser.add_argument("--equilibrium-qrte-dir", type=Path, default=DEFAULT_EQUILIBRIUM_QRTE_DIR)
    parser.add_argument("--equilibrium-qkud-dir", type=Path, default=DEFAULT_EQUILIBRIUM_QKUD_DIR)
    parser.add_argument("--equilibrium-gap-dir", type=Path, default=DEFAULT_EQUILIBRIUM_GAP_DIR)
    parser.add_argument("--equilibrium-qrte-rerun-dir", type=Path, default=DEFAULT_EQUILIBRIUM_QRTE_RERUN_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--expected-count", type=int, default=16)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--no-equilibrium", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(
        qrte_dir=args.qrte_dir,
        qkud_dir=args.qkud_dir,
        gap_dir=args.gap_dir,
        equilibrium_qrte_dir=args.equilibrium_qrte_dir,
        equilibrium_qkud_dir=args.equilibrium_qkud_dir,
        equilibrium_gap_dir=args.equilibrium_gap_dir,
        equilibrium_qrte_rerun_dir=args.equilibrium_qrte_rerun_dir,
        include_equilibrium=not args.no_equilibrium,
        expected_count=int(args.expected_count),
        allow_missing=bool(args.allow_missing),
    )
    if not rows:
        raise SystemExit("No N2 PES overlap/gap rows found.")

    csv_path = args.output_prefix.with_suffix(".csv")
    _write_rows(rows, csv_path)
    _plot(rows, args.output_prefix)

    hit_rows = _hit_rows(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")
    pes_points = len({row["case_id"] for row in rows if row.get("point_kind") == "pes"})
    equilibrium_points = len({row["case_id"] for row in rows if row.get("point_kind") == "equilibrium"})
    print(
        f"Included {pes_points} N2 PES points, {equilibrium_points} equilibrium points, "
        f"and {len(hit_rows)} method-hit points."
    )


if __name__ == "__main__":
    main()
