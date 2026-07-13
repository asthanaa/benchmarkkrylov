#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot N2 bond distance versus two-body cumulant norm squared."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "benchmarkkrylov-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = BENCHMARK_ROOT / "runs" / "n2_pes_qrte_qkud_cumulants.csv"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "n2_pes_2body_cumulant_norm2_vs_bond_distance"
EQUILIBRIUM_CUMULANT_BY_SPIN = {
    0: BENCHMARK_ROOT
    / "runs"
    / "two_body_cumulant_norms"
    / "qkud_eps0p05_converged"
    / "N2_equi_cc_pVTZ_0"
    / "cumulant_2body.json",
    2: BENCHMARK_ROOT
    / "runs"
    / "two_body_cumulant_norms"
    / "qkud_eps0p05_converged"
    / "N2_equi_cc_pVTZ_2"
    / "cumulant_2body.json",
}


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _read_rows(path: Path, *, include_equilibrium: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            norm2 = _float_or_none(row.get("cumulant_2rdm_frobenius_norm_squared"))
            if norm2 is None:
                continue
            rows.append(
                {
                    "bond_length_angstrom": float(row["bond_length_angstrom"]),
                    "spin": int(row["spin"]),
                    "norm2": norm2,
                    "case_id": row["case_id"],
                    "source": "PES",
                }
            )

    if include_equilibrium:
        for spin, source in EQUILIBRIUM_CUMULANT_BY_SPIN.items():
            if not source.exists():
                continue
            with source.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            norm2 = _float_or_none(payload.get("cumulant_2rdm_frobenius_norm_squared"))
            if norm2 is None:
                continue
            rows.append(
                {
                    "bond_length_angstrom": 1.20,
                    "spin": int(spin),
                    "norm2": norm2,
                    "case_id": f"N2_equi_cc_pVTZ_{spin}",
                    "source": "equilibrium",
                }
            )

    rows.sort(key=lambda item: (item["spin"], item["bond_length_angstrom"]))
    return rows


def plot(rows: list[dict[str, Any]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    colors = {0: "#2563eb", 2: "#c026d3"}
    markers = {0: "o", 2: "s"}
    labels = {0: "N2_0", 2: "N2_2"}

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    for spin in (0, 2):
        spin_rows = [row for row in rows if row["spin"] == spin]
        if not spin_rows:
            continue
        x_values = [row["bond_length_angstrom"] for row in spin_rows]
        y_values = [row["norm2"] for row in spin_rows]
        ax.plot(
            x_values,
            y_values,
            color=colors[spin],
            linewidth=1.9,
            alpha=0.85,
            zorder=2,
        )
        ax.scatter(
            x_values,
            y_values,
            marker=markers[spin],
            s=58,
            color=colors[spin],
            edgecolors="white",
            linewidths=0.8,
            label=labels[spin],
            zorder=3,
        )

    ax.set_title("N2 PES: 2-body cumulant norm squared vs bond distance")
    ax.set_xlabel("N-N bond distance (Angstrom)")
    ax.set_ylabel("2-body cumulant 2-RDM Frobenius norm squared")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--include-equilibrium", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input_csv, include_equilibrium=bool(args.include_equilibrium))
    if not rows:
        raise SystemExit(f"No plottable rows in {args.input_csv}")
    plot(rows, args.output_prefix)
    print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    print(f"Wrote {args.output_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
