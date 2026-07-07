#!/usr/bin/env python
"""Plot ADAPT, QKUD eps=0.1, and QKUD eps=0.2 for eps=0.2-converged cases."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BENCHMARK_ROOT / "scripts"))

import collect_and_plot as base_plot  # noqa: E402


DEFAULT_EPS0P1_CSV = BENCHMARK_ROOT / "plots" / "adapt_vs_qkud_iterations_vs_entropy.csv"
DEFAULT_EPS0P2_CSV = (
    BENCHMARK_ROOT
    / "plots"
    / "mushir_dt0p2_eps0p2"
    / "adapt_vs_qkud_eps0p2_iterations_vs_entropy.csv"
)
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "adapt_vs_qkud_eps0p1_eps0p2_iterations_vs_entropy"

FIELDNAMES = [
    "case_id",
    "source_name",
    "h0.25",
    "adapt_chem_accuracy_iteration",
    "qkud_eps0p1_chem_accuracy_iteration",
    "qkud_eps0p1_hit_chemical_accuracy",
    "qkud_eps0p1_status",
    "qkud_eps0p2_chem_accuracy_iteration",
    "qkud_eps0p2_hit_chemical_accuracy",
    "qkud_eps0p2_status",
    "casscf_energy_hartree",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_method_hit(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    return (
        row.get("method_status") == "completed"
        and row.get("method_hit_chemical_accuracy") == "true"
        and bool(row.get("method_chem_accuracy_iteration"))
    )


def _positive_number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number > 0.0 and math.isfinite(number):
        return number
    return None


def _build_rows(eps0p1_rows: list[dict[str, str]], eps0p2_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    eps0p1_by_case = {row["case_id"]: row for row in eps0p1_rows}
    rows: list[dict[str, Any]] = []
    for eps0p2_row in eps0p2_rows:
        if not _is_method_hit(eps0p2_row):
            continue

        case_id = eps0p2_row["case_id"]
        eps0p1_row = eps0p1_by_case.get(case_id)
        adapt_iteration = eps0p2_row.get("adapt_chem_accuracy_iteration") or (
            eps0p1_row.get("adapt_chem_accuracy_iteration", "") if eps0p1_row else ""
        )
        if _positive_number(adapt_iteration) is None:
            continue

        rows.append(
            {
                "case_id": case_id,
                "source_name": eps0p2_row["source_name"],
                "h0.25": float(eps0p2_row["h0.25"]),
                "adapt_chem_accuracy_iteration": adapt_iteration,
                "qkud_eps0p1_chem_accuracy_iteration": (
                    eps0p1_row.get("method_chem_accuracy_iteration", "") if _is_method_hit(eps0p1_row) else ""
                ),
                "qkud_eps0p1_hit_chemical_accuracy": str(_is_method_hit(eps0p1_row)).lower(),
                "qkud_eps0p1_status": eps0p1_row.get("method_status", "missing") if eps0p1_row else "missing",
                "qkud_eps0p2_chem_accuracy_iteration": eps0p2_row["method_chem_accuracy_iteration"],
                "qkud_eps0p2_hit_chemical_accuracy": "true",
                "qkud_eps0p2_status": eps0p2_row["method_status"],
                "casscf_energy_hartree": float(eps0p2_row["casscf_energy_hartree"]),
            }
        )
    rows.sort(key=lambda row: float(row["h0.25"]))
    return rows


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _finite_xy(rows: list[dict[str, Any]], y_key: str) -> tuple[Any, Any]:
    points: list[tuple[float, float]] = []
    for row in rows:
        y = _positive_number(str(row.get(y_key, "")))
        if y is not None:
            points.append((float(row["h0.25"]), y))
    points.sort(key=lambda item: item[0])
    if not points:
        return base_plot.np.asarray([], dtype=float), base_plot.np.asarray([], dtype=float)
    x, y = zip(*points)
    return base_plot.np.asarray(x, dtype=float), base_plot.np.asarray(y, dtype=float)


def _plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
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
            "legend.fontsize": 9,
        }
    )

    series = [
        ("ADAPT", "adapt_chem_accuracy_iteration", "#0072B2", "o", 48),
        ("QKUD eps=0.1", "qkud_eps0p1_chem_accuracy_iteration", "#D55E00", "^", 58),
        ("QKUD eps=0.2", "qkud_eps0p2_chem_accuracy_iteration", "#009E73", "D", 54),
    ]

    fig, ax = base_plot.plt.subplots(figsize=(7.4, 4.9))
    for label, key, color, marker, size in series:
        x, y = _finite_xy(rows, key)
        if not x.size:
            continue
        ax.scatter(x, y, s=size, color=color, marker=marker, label=label, zorder=3)
        fit = base_plot._fit_semilog(x, y)
        if fit is not None:
            x_fit, y_fit, r2 = fit
            ax.plot(
                x_fit,
                y_fit,
                color=color,
                linewidth=1.6,
                linestyle="--",
                alpha=0.72,
                label=f"{label} fit R2={r2:.3f}",
            )

    ax.set_yscale("log")
    ax.set_xlabel("Renyi entropy h0.25")
    ax.set_ylabel("Iterations to chemical accuracy")
    ax.set_title(f"ADAPT vs QKUD on eps=0.2-converged systems (n={len(rows)})")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    base_plot.plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps0p1-csv", type=Path, default=DEFAULT_EPS0P1_CSV)
    parser.add_argument("--eps0p2-csv", type=Path, default=DEFAULT_EPS0P2_CSV)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(_read_rows(args.eps0p1_csv), _read_rows(args.eps0p2_csv))
    if not rows:
        raise SystemExit("No eps=0.2 QKUD chemical-accuracy hits found.")

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    _write_rows(rows, csv_path)
    _plot(rows, png_path, pdf_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Included {len(rows)} QKUD eps=0.2-converged systems.")


if __name__ == "__main__":
    main()
