#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot N2 PES Krylov iterations versus two-body cumulant norm squared."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = BENCHMARK_ROOT / "runs" / "n2_pes_qrte_qkud_cumulants.csv"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "n2_pes_iterations_vs_2body_cumulant_norm2"

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 450,
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.85,
    }
)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            norm2 = _float_or_none(row.get("cumulant_2rdm_frobenius_norm_squared"))
            if norm2 is None:
                continue
            rows.append(
                {
                    **row,
                    "bond_length_angstrom": float(row["bond_length_angstrom"]),
                    "spin": int(row["spin"]),
                    "cumulant_2rdm_frobenius_norm_squared": norm2,
                    "qrte_hit_iteration": _float_or_none(row.get("qrte_hit_iteration")),
                    "qrte_steps_completed": _float_or_none(row.get("qrte_steps_completed")),
                    "qrte_hit_chemical_accuracy": _boolish(row.get("qrte_hit_chemical_accuracy")),
                    "qkud_hit_iteration": _float_or_none(row.get("qkud_hit_iteration")),
                    "qkud_steps_completed": _float_or_none(row.get("qkud_steps_completed")),
                    "qkud_hit_chemical_accuracy": _boolish(row.get("qkud_hit_chemical_accuracy")),
                }
            )
    rows.sort(key=lambda item: (item["spin"], item["bond_length_angstrom"]))
    return rows


def _format_bond(value: float) -> str:
    return f"{float(value):.2f}"


def _plot_method(
    ax,
    rows: list[dict[str, Any]],
    *,
    method: str,
    title: str,
    cmap,
    norm: mcolors.Normalize,
) -> bool:
    labels = {0: r"N$_2$, spin 0", 2: r"N$_2$, spin 2"}
    markers = {0: "o", 2: "s"}
    all_y_values: list[float] = []
    saw_miss = False
    for spin in (0, 2):
        spin_rows = [row for row in rows if int(row["spin"]) == spin]
        hits = [row for row in spin_rows if row[f"{method}_hit_chemical_accuracy"]]
        misses = [row for row in spin_rows if not row[f"{method}_hit_chemical_accuracy"]]
        hit_x = [row["cumulant_2rdm_frobenius_norm_squared"] for row in hits]
        hit_y = [row[f"{method}_hit_iteration"] for row in hits]
        hit_bonds = [row["bond_length_angstrom"] for row in hits]
        miss_x = [row["cumulant_2rdm_frobenius_norm_squared"] for row in misses]
        miss_y = [row[f"{method}_steps_completed"] for row in misses]
        miss_bonds = [row["bond_length_angstrom"] for row in misses]
        all_y_values.extend(float(value) for value in [*hit_y, *miss_y] if value is not None)
        saw_miss = saw_miss or bool(misses)

        if hits:
            ax.scatter(
                hit_x,
                hit_y,
                c=hit_bonds,
                cmap=cmap,
                norm=norm,
                marker=markers[spin],
                edgecolors="white",
                linewidths=0.9,
                s=78,
                label=labels[spin],
                zorder=3,
            )
        if misses:
            ax.scatter(
                miss_x,
                miss_y,
                marker=markers[spin],
                facecolors="none",
                edgecolors=cmap(norm(miss_bonds)),
                linewidths=2.0,
                s=96,
                label=f"{labels[spin]} no hit",
                zorder=4,
            )

    if all_y_values:
        if method == "qkud":
            y_min = max(0.0, min(all_y_values) - 0.6)
            y_max = max(all_y_values) + 1.1
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(range(math.ceil(y_min), math.floor(y_max) + 1))
        else:
            y_max = max(all_y_values) + 6.0
            ax.set_ylim(0.0, y_max)

    ax.set_title(title)
    ax.set_xlabel(r"2-body cumulant norm squared, $\|{}^{2}\Delta\|_F^2$")
    ax.set_xlim(0.0, 11.6)
    ax.grid(True, color="#e5e7eb", linewidth=0.75, alpha=0.9)
    ax.grid(True, which="minor", color="#f3f4f6", linewidth=0.55, alpha=0.85)
    ax.minorticks_on()
    ax.tick_params(direction="out", length=4, width=0.8, colors="#111827")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return saw_miss


def plot(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    bond_values = [float(row["bond_length_angstrom"]) for row in rows]
    unique_bonds = sorted(set(bond_values))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=min(bond_values), vmax=max(bond_values))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharex=True, constrained_layout=True)
    qrte_saw_miss = _plot_method(
        axes[0],
        rows,
        method="qrte",
        title=r"QRTE, $\Delta t=0.05$",
        cmap=cmap,
        norm=norm,
    )
    _plot_method(
        axes[1],
        rows,
        method="qkud",
        title=r"QKUD, $\epsilon=0.05$",
        cmap=cmap,
        norm=norm,
    )
    for ax in axes:
        ax.set_ylabel("Iterations to chemical accuracy")
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=7.5,
            label=r"N$_2$, spin 0",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#4b5563",
            markeredgecolor="white",
            markeredgewidth=0.9,
            markersize=7.5,
            label=r"N$_2$, spin 2",
        ),
    ]
    if qrte_saw_miss:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="none",
                markeredgecolor="#4b5563",
                markeredgewidth=1.8,
                markersize=8.0,
                label="no QRTE hit",
            )
        )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.46, -0.105),
        ncol=len(legend_handles),
        handletextpad=0.55,
        columnspacing=1.4,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=axes, shrink=0.92, pad=0.018, aspect=28)
    colorbar.set_label(r"N-N bond distance, $R$ ($\AA$)")
    colorbar.set_ticks(unique_bonds)
    colorbar.set_ticklabels([_format_bond(value) for value in unique_bonds])
    colorbar.outline.set_linewidth(0.65)

    fig.savefig(output_prefix.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No plottable rows in {args.input_csv}")
    plot(rows, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
