#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Collect sector gaps and plot Krylov iterations versus the fixed-sector gap."""

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
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_GAP_DIR = BENCHMARK_ROOT / "runs" / "initial_state_gap"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "sector_gap_vs_krylov_iterations"

BASE_QKUD_DIR = BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.05"
BASE_QRTE_DIR = BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.05"

EXCLUDED_CASE_IDS = {
    "Ferrocene_ceo",
    "feo4_2minus_ceo",
}

QKUD_RERUN_SUMMARIES = {
    "cyclobutadiene_ceo": BENCHMARK_ROOT
    / "runs"
    / "qkud_corrected_active_space_eps0.05"
    / "cyclobutadiene_ceo"
    / "summary.json",
    "m_Benzyne_equi_cc_pVTZ_0": BENCHMARK_ROOT
    / "runs"
    / "qkud_corrected_active_space_eps0.05"
    / "m_Benzyne_equi_cc_pVTZ_0"
    / "summary.json",
    "m_Benzyne_equi_cc_pVTZ_2": BENCHMARK_ROOT
    / "runs"
    / "qkud_corrected_active_space_eps0.05"
    / "m_Benzyne_equi_cc_pVTZ_2"
    / "summary.json",
    "FeO": BENCHMARK_ROOT / "runs" / "qkud_fe_eps0.05" / "FeO" / "summary.json",
    "feno_2plus": BENCHMARK_ROOT / "runs" / "qkud_fe_eps0.05" / "feno_2plus" / "summary.json",
    "fenoco3_minus": BENCHMARK_ROOT / "runs" / "qkud_fe_eps0.05" / "fenoco3_minus" / "summary.json",
}

QRTE_RERUN_SUMMARIES = {
    "cyclobutadiene_ceo": BENCHMARK_ROOT
    / "runs"
    / "qrte_corrected_active_space_dt0.05"
    / "cyclobutadiene_ceo"
    / "summary.json",
    "m_Benzyne_equi_cc_pVTZ_0": BENCHMARK_ROOT
    / "runs"
    / "qrte_mbenzyne_local_casscf_natorb_dt0.05"
    / "m_Benzyne_equi_cc_pVTZ_0"
    / "summary.json",
    "m_Benzyne_equi_cc_pVTZ_2": BENCHMARK_ROOT
    / "runs"
    / "qrte_mbenzyne_local_casscf_natorb_dt0.05"
    / "m_Benzyne_equi_cc_pVTZ_2"
    / "summary.json",
    "fenoco3_minus": BENCHMARK_ROOT
    / "runs"
    / "qrte_sfx2c1erohf_natorb_unconverged_dt0.05_100iter"
    / "fenoco3_minus"
    / "summary.json",
    "N2_equi_cc_pVTZ_2": BENCHMARK_ROOT
    / "runs"
    / "qrte_natorb_dt0.05_100iter"
    / "N2_equi_cc_pVTZ_2"
    / "summary.json",
    "FeO": BENCHMARK_ROOT
    / "runs"
    / "qrte_rohf_molvqe_fe_cases_100iter"
    / "FeO"
    / "dt0p05"
    / "summary.json",
    "feno_2plus": BENCHMARK_ROOT
    / "runs"
    / "qrte_sfx2c1erohf_natorb_dt0.05_100iter"
    / "feno_2plus"
    / "summary.json",
}

FIELDNAMES = [
    "case_id",
    "source_name",
    "h0.25",
    "gap_hartree",
    "gap_millihartree",
    "gap_ev",
    "sector_ground_energy_hartree",
    "first_excited_energy_hartree",
    "overlap_weight",
    "negative_log10_overlap",
    "sector_dim",
    "n_alpha",
    "n_beta",
    "gap_path",
    "qkud_eps0p05_chem_accuracy_iteration",
    "qkud_hit_chemical_accuracy",
    "qkud_summary_path",
    "qrte_dt0p05_chem_accuracy_iteration",
    "qrte_hit_chemical_accuracy",
    "qrte_summary_path",
    "qkud_only_hit",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_for_case(
    case_id: str,
    base_dir: Path,
    rerun_summaries: dict[str, Path],
) -> tuple[Path, dict[str, Any] | None]:
    rerun_path = rerun_summaries.get(case_id)
    if rerun_path is not None and rerun_path.exists():
        return rerun_path, _load_json(rerun_path)
    base_path = base_dir / case_id / "summary.json"
    if base_path.exists():
        return base_path, _load_json(base_path)
    return base_path, None


def _relative_path(path: Path) -> str:
    if not path.exists():
        return ""
    return str(path.relative_to(BENCHMARK_ROOT))


def _summary_hit(summary: dict[str, Any] | None) -> bool:
    return bool(
        summary is not None
        and summary.get("status") == "completed"
        and summary.get("hit_chemical_accuracy") is True
        and summary.get("hit_iteration") is not None
    )


def _hit_iteration(summary: dict[str, Any] | None) -> float | str:
    if not _summary_hit(summary):
        return ""
    return float(summary["hit_iteration"])


def _target_manifest_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in manifest_rows if row["case_id"] not in EXCLUDED_CASE_IDS]


def _valid_gap(gap: dict[str, Any]) -> bool:
    if gap.get("status") != "completed":
        return False
    try:
        value = float(gap["gap_hartree"])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(value > 0.0 and math.isfinite(value))


def _build_rows(manifest_rows: list[dict[str, str]], gap_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing_or_invalid: list[str] = []
    for manifest_row in _target_manifest_rows(manifest_rows):
        case_id = manifest_row["case_id"]
        gap_path = gap_dir / case_id / "sector_gap.json"
        if not gap_path.exists():
            missing_or_invalid.append(case_id)
            continue
        gap = _load_json(gap_path)
        if not _valid_gap(gap):
            missing_or_invalid.append(case_id)
            continue

        qkud_path, qkud_summary = _summary_for_case(case_id, BASE_QKUD_DIR, QKUD_RERUN_SUMMARIES)
        qrte_path, qrte_summary = _summary_for_case(case_id, BASE_QRTE_DIR, QRTE_RERUN_SUMMARIES)
        qkud_hit = _summary_hit(qkud_summary)
        qrte_hit = _summary_hit(qrte_summary)
        rows.append(
            {
                "case_id": case_id,
                "source_name": manifest_row["source_name"],
                "h0.25": float(manifest_row["h0.25"]),
                "gap_hartree": float(gap["gap_hartree"]),
                "gap_millihartree": float(gap["gap_millihartree"]),
                "gap_ev": float(gap["gap_ev"]),
                "sector_ground_energy_hartree": float(gap["sector_ground_energy_hartree"]),
                "first_excited_energy_hartree": float(gap["first_excited_energy_hartree"]),
                "overlap_weight": float(gap["overlap_weight"]),
                "negative_log10_overlap": float(gap["negative_log10_overlap"]),
                "sector_dim": int(gap["sector_dim"]),
                "n_alpha": int(gap["n_alpha"]),
                "n_beta": int(gap["n_beta"]),
                "gap_path": str(gap_path.relative_to(BENCHMARK_ROOT)),
                "qkud_eps0p05_chem_accuracy_iteration": _hit_iteration(qkud_summary),
                "qkud_hit_chemical_accuracy": str(qkud_hit).lower(),
                "qkud_summary_path": _relative_path(qkud_path),
                "qrte_dt0p05_chem_accuracy_iteration": _hit_iteration(qrte_summary),
                "qrte_hit_chemical_accuracy": str(qrte_hit).lower(),
                "qrte_summary_path": _relative_path(qrte_path),
                "qkud_only_hit": str(qkud_hit and not qrte_hit).lower(),
            }
        )
    rows.sort(key=lambda row: (float(row["gap_ev"]), row["case_id"]))
    return rows, missing_or_invalid


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _positive_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if number > 0.0 and math.isfinite(number):
        return number
    return None


def _finite_xy(rows: list[dict[str, Any]], iteration_key: str, hit_key: str) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row[hit_key] != "true":
            continue
        x = _positive_number(row["gap_ev"])
        y = _positive_number(row[iteration_key])
        if x is not None and y is not None:
            points.append((x, y))
    points.sort(key=lambda item: item[0])
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x_values, y_values = zip(*points)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def _plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path, *, xscale: str) -> None:
    qkud_hit_rows = [row for row in rows if row["qkud_hit_chemical_accuracy"] == "true"]
    qrte_hit_rows = [row for row in rows if row["qrte_hit_chemical_accuracy"] == "true"]
    qkud_only_rows = [row for row in rows if row["qkud_only_hit"] == "true"]

    qkud_x, qkud_y = _finite_xy(rows, "qkud_eps0p05_chem_accuracy_iteration", "qkud_hit_chemical_accuracy")
    qrte_x, qrte_y = _finite_xy(rows, "qrte_dt0p05_chem_accuracy_iteration", "qrte_hit_chemical_accuracy")

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

    fig, ax = plt.subplots(figsize=(7.1, 4.65), constrained_layout=True)
    ax.scatter(
        qrte_x,
        qrte_y,
        s=54,
        color="#D55E00",
        marker="s",
        label=rf"QRTE $\Delta t=0.05$ (n={len(qrte_hit_rows)})",
        zorder=4,
    )
    ax.scatter(
        qkud_x,
        qkud_y,
        s=58,
        color="#009E73",
        marker="^",
        label=rf"QKUD $\epsilon=0.05$ (n={len(qkud_hit_rows)})",
        zorder=5,
    )

    if qkud_only_rows:
        qkud_only_x = np.asarray([float(row["gap_ev"]) for row in qkud_only_rows], dtype=float)
        qkud_only_y = np.asarray([float(row["qkud_eps0p05_chem_accuracy_iteration"]) for row in qkud_only_rows], dtype=float)
        ax.scatter(
            qkud_only_x,
            qkud_only_y,
            s=150,
            color="#F0E442",
            edgecolor="black",
            linewidth=0.7,
            marker="*",
            label="QKUD hit; QRTE not hit",
            zorder=6,
        )
        for row in qkud_only_rows:
            ax.annotate(
                row["case_id"],
                (float(row["gap_ev"]), float(row["qkud_eps0p05_chem_accuracy_iteration"])),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.8,
                color="#333333",
                ha="left",
                va="bottom",
            )

    for row in qrte_hit_rows:
        y_value = _positive_number(row["qrte_dt0p05_chem_accuracy_iteration"])
        if y_value is None or y_value < 50.0:
            continue
        ax.annotate(
            row["case_id"],
            (float(row["gap_ev"]), y_value),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.8,
            color="#333333",
            ha="left",
            va="bottom",
        )

    ax.set_xscale(xscale)
    ax.set_yscale("log")
    if xscale == "log":
        ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Fixed-sector gap, $E_1 - E_0$ (eV)")
    ax.set_ylabel("Iterations to chemical accuracy")
    ax.grid(True, which="major", color="#D8D8D8", alpha=0.55, linewidth=0.7)
    ax.grid(True, which="minor", color="#E8E8E8", alpha=0.35, linewidth=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), handlelength=1.6)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gap-dir", type=Path, default=DEFAULT_GAP_DIR)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--xscale", choices=("linear", "log"), default="log")
    parser.add_argument("--expected-target-count", type=int, default=27)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = _read_manifest(args.manifest)
    target_rows = _target_manifest_rows(manifest_rows)
    if len(target_rows) != int(args.expected_target_count):
        raise SystemExit(f"Expected {args.expected_target_count} target manifest rows, found {len(target_rows)}")

    rows, missing = _build_rows(manifest_rows, args.gap_dir)
    if missing and not args.allow_missing:
        missing_text = "\n  ".join(missing)
        raise SystemExit(
            f"Missing or invalid sector gap JSON for {len(missing)} target systems:\n  {missing_text}\n"
            "Rerun with --allow-missing for a preview plot."
        )
    if not rows:
        raise SystemExit("No valid sector gap rows found.")

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    _write_rows(rows, csv_path)
    _plot(rows, png_path, pdf_path, xscale=str(args.xscale))

    qkud_hits = [row for row in rows if row["qkud_hit_chemical_accuracy"] == "true"]
    qrte_hits = [row for row in rows if row["qrte_hit_chemical_accuracy"] == "true"]
    qkud_only = [row["case_id"] for row in rows if row["qkud_only_hit"] == "true"]
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Included {len(rows)} sector-gap rows out of {len(target_rows)} target systems.")
    print(f"QKUD hits with gap: {len(qkud_hits)}")
    print(f"QRTE hits with gap: {len(qrte_hits)}")
    print(f"Highlighted QKUD-only hits: {', '.join(qkud_only) if qkud_only else 'none'}")
    if missing:
        print(f"Missing or invalid sector-gap rows skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()
