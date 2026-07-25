#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Plot QKUD eps=0.05, QRTE dt=0.05, and ADAPT iterations versus Renyi entropy."""

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


BENCHMARK_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = BENCHMARK_ROOT / "manifests" / "molvqe21_fig1b_manifest.csv"
DEFAULT_OUTPUT_PREFIX = BENCHMARK_ROOT / "plots" / "qkud_eps0p05_vs_qrte_dt0p05_iterations_vs_entropy"

BASE_QKUD_DIR = BENCHMARK_ROOT / "runs" / "qkud_natorb_eps0.05"
BASE_QRTE_DIR = BENCHMARK_ROOT / "runs" / "qrte_natorb_dt0.05"

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

ADAPT_FIXED_OVERRIDES = {
    "H10_near_equi_sto3g_0": {
        "iteration": 324.5,
        "source": "paper_fig1_extrapolated",
        "note": "H10 ADAPT point from arXiv:2603.13073v2 Appendix C: n_ADAPT=324.5 +/- 2.1.",
    },
}

FIELDNAMES = [
    "case_id",
    "source_name",
    "h0.25",
    "adapt_chem_accuracy_iteration",
    "adapt_hit_chemical_accuracy",
    "adapt_source",
    "adapt_note",
    "qkud_eps0p05_chem_accuracy_iteration",
    "qkud_hit_chemical_accuracy",
    "qkud_status",
    "qkud_n_steps_completed",
    "qkud_n_steps_requested",
    "qkud_summary_path",
    "qkud_result_source",
    "qrte_dt0p05_chem_accuracy_iteration",
    "qrte_hit_chemical_accuracy",
    "qrte_status",
    "qrte_n_steps_completed",
    "qrte_n_steps_requested",
    "qrte_summary_path",
    "qrte_result_source",
    "qkud_only_hit",
    "casscf_energy_hartree",
]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_hit(summary: dict[str, Any] | None) -> bool:
    return bool(
        summary is not None
        and summary.get("status") == "completed"
        and summary.get("hit_chemical_accuracy") is True
        and summary.get("hit_iteration") is not None
    )


def _summary_for_case(
    case_id: str,
    base_dir: Path,
    rerun_summaries: dict[str, Path],
) -> tuple[Path, str, dict[str, Any] | None]:
    rerun_path = rerun_summaries.get(case_id)
    if rerun_path is not None and rerun_path.exists():
        return rerun_path, "rerun", _load_json(rerun_path)
    base_path = base_dir / case_id / "summary.json"
    return base_path, "base", _load_json(base_path)


def _summary_path_text(path: Path) -> str:
    if not path.exists():
        return ""
    return str(path.relative_to(BENCHMARK_ROOT))


def _adapt_iteration(row: dict[str, str]) -> tuple[float | str, bool, str, str]:
    override = ADAPT_FIXED_OVERRIDES.get(row["case_id"])
    if override is not None:
        return float(override["iteration"]), True, str(override["source"]), str(override["note"])

    value = row.get("adapt_chem_accuracy_iteration", "")
    if row.get("has_adapt_chem_hit") == "true" and value:
        return float(value), True, "manifest", ""
    return "", False, "", ""


def _iteration(summary: dict[str, Any] | None) -> float | str:
    if not _summary_hit(summary):
        return ""
    return float(summary["hit_iteration"])


def _steps_completed(summary: dict[str, Any] | None) -> int | str:
    if summary is None:
        return ""
    return int(summary.get("n_steps_completed", summary.get("n_steps", 0)) or 0)


def _steps_requested(summary: dict[str, Any] | None) -> int | str:
    if summary is None:
        return ""
    return int(summary.get("n_steps_requested", 0) or 0)


def _status(summary: dict[str, Any] | None) -> str:
    if summary is None:
        return "missing"
    return str(summary.get("status", "unknown"))


def _build_rows(manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        case_id = manifest_row["case_id"]
        adapt_iteration, adapt_hit, adapt_source, adapt_note = _adapt_iteration(manifest_row)
        qkud_path, qkud_source, qkud_summary = _summary_for_case(case_id, BASE_QKUD_DIR, QKUD_RERUN_SUMMARIES)
        qrte_path, qrte_source, qrte_summary = _summary_for_case(case_id, BASE_QRTE_DIR, QRTE_RERUN_SUMMARIES)

        qkud_hit = _summary_hit(qkud_summary)
        qrte_hit = _summary_hit(qrte_summary)
        rows.append(
            {
                "case_id": case_id,
                "source_name": manifest_row["source_name"],
                "h0.25": float(manifest_row["h0.25"]),
                "adapt_chem_accuracy_iteration": adapt_iteration,
                "adapt_hit_chemical_accuracy": str(adapt_hit).lower(),
                "adapt_source": adapt_source,
                "adapt_note": adapt_note,
                "qkud_eps0p05_chem_accuracy_iteration": _iteration(qkud_summary),
                "qkud_hit_chemical_accuracy": str(qkud_hit).lower(),
                "qkud_status": _status(qkud_summary),
                "qkud_n_steps_completed": _steps_completed(qkud_summary),
                "qkud_n_steps_requested": _steps_requested(qkud_summary),
                "qkud_summary_path": _summary_path_text(qkud_path),
                "qkud_result_source": qkud_source if qkud_summary is not None else "missing",
                "qrte_dt0p05_chem_accuracy_iteration": _iteration(qrte_summary),
                "qrte_hit_chemical_accuracy": str(qrte_hit).lower(),
                "qrte_status": _status(qrte_summary),
                "qrte_n_steps_completed": _steps_completed(qrte_summary),
                "qrte_n_steps_requested": _steps_requested(qrte_summary),
                "qrte_summary_path": _summary_path_text(qrte_path),
                "qrte_result_source": qrte_source if qrte_summary is not None else "missing",
                "qkud_only_hit": str(qkud_hit and not qrte_hit).lower(),
                "casscf_energy_hartree": float(manifest_row["casscf_energy_hartree"]),
            }
        )
    rows.sort(key=lambda row: (float(row["h0.25"]), row["case_id"]))
    return rows


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


def _finite_xy(rows: list[dict[str, Any]], y_key: str, *, row_filter: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    for row in rows:
        if row_filter is not None and not row_filter(row):
            continue
        y = _positive_number(row.get(y_key))
        if y is not None:
            points.append((float(row["h0.25"]), y))
    points.sort(key=lambda item: item[0])
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _fit_semilog(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
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
    return x_fit, y_fit, float(r2)


def _plot(rows: list[dict[str, Any]], png_path: Path, pdf_path: Path) -> None:
    qkud_hit_rows = [row for row in rows if row["qkud_hit_chemical_accuracy"] == "true"]
    qrte_hit_rows = [row for row in rows if row["qrte_hit_chemical_accuracy"] == "true"]
    qkud_only_rows = [row for row in rows if row["qkud_only_hit"] == "true"]
    qkud_hit_case_ids = {str(row["case_id"]) for row in qkud_hit_rows}
    qkud_only_case_ids = {str(row["case_id"]) for row in qkud_only_rows}
    adapt_hit_rows = [
        row
        for row in rows
        if row["adapt_hit_chemical_accuracy"] == "true"
        and str(row["case_id"]) in qkud_hit_case_ids
    ]
    qkud_marker_rows = [
        row
        for row in qkud_hit_rows
        if str(row["case_id"]) not in qkud_only_case_ids
    ]

    adapt_x, adapt_y = _finite_xy(adapt_hit_rows, "adapt_chem_accuracy_iteration")
    qrte_x, qrte_y = _finite_xy(qrte_hit_rows, "qrte_dt0p05_chem_accuracy_iteration")
    qkud_x, qkud_y = _finite_xy(qkud_hit_rows, "qkud_eps0p05_chem_accuracy_iteration")
    qkud_marker_x, qkud_marker_y = _finite_xy(qkud_marker_rows, "qkud_eps0p05_chem_accuracy_iteration")
    qkud_only_x, qkud_only_y = _finite_xy(qkud_only_rows, "qkud_eps0p05_chem_accuracy_iteration")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.1,
            "axes.labelsize": 16,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 12.2,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "xtick.major.size": 5.0,
            "ytick.major.size": 5.0,
            "xtick.minor.size": 2.9,
            "ytick.minor.size": 2.9,
        }
    )

    qrte_color = "#0072B2"
    qkud_color = "#D55E00"
    adapt_face = "#D7D7D7"
    adapt_edge = "#8B8B8B"

    fig, ax = plt.subplots(figsize=(8.0, 5.25), constrained_layout=True)
    ax.scatter(
        adapt_x,
        adapt_y,
        s=56,
        facecolor=adapt_face,
        edgecolor=adapt_edge,
        linewidth=0.55,
        alpha=0.82,
        marker="o",
        label="ADAPT-VQE",
        zorder=2,
    )
    ax.scatter(
        qrte_x,
        qrte_y,
        s=74,
        color=qrte_color,
        marker="s",
        label="QRTE",
        zorder=4,
    )
    ax.scatter(
        qkud_marker_x,
        qkud_marker_y,
        s=80,
        color=qkud_color,
        marker="^",
        label="QKUD",
        zorder=5,
    )

    if qkud_only_x.size:
        ax.scatter(
            qkud_only_x,
            qkud_only_y,
            s=220,
            color="#F0E442",
            edgecolor="black",
            linewidth=0.9,
            marker="*",
            label="_nolegend_",
            zorder=6,
        )

    for row in rows:
        qkud_iter = _positive_number(row["qkud_eps0p05_chem_accuracy_iteration"])
        qrte_iter = _positive_number(row["qrte_dt0p05_chem_accuracy_iteration"])
        if qkud_iter is None or qrte_iter is None:
            continue
        x = float(row["h0.25"])
        ax.plot([x, x], [qrte_iter, qkud_iter], color="#999999", alpha=0.24, linewidth=0.9, zorder=1)

    for label, x, y, color, linestyle, linewidth, alpha in (
        ("ADAPT-VQE", adapt_x, adapt_y, "#A8A8A8", "--", 1.65, 0.78),
        ("QRTE", qrte_x, qrte_y, qrte_color, "-", 2.1, 0.82),
        ("QKUD", qkud_x, qkud_y, qkud_color, "-", 2.1, 0.82),
    ):
        fit = _fit_semilog(x, y)
        if fit is None:
            continue
        x_fit, y_fit, r2 = fit
        ax.plot(
            x_fit,
            y_fit,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=rf"{label} fit $R^2$={r2:.3f}",
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Renyi entropy, $h_{0.25}$")
    ax.set_ylabel("Iterations to chemical accuracy")
    ax.grid(True, which="major", color="#D8D8D8", alpha=0.55, linewidth=0.75)
    ax.grid(True, which="minor", color="#E8E8E8", alpha=0.35, linewidth=0.48)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), handlelength=1.65)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--expected-qkud-hits", type=int, default=27)
    parser.add_argument("--expected-qrte-hits", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _build_rows(_read_manifest(args.manifest))
    adapt_hits = [row for row in rows if row["adapt_hit_chemical_accuracy"] == "true"]
    qkud_hits = [row for row in rows if row["qkud_hit_chemical_accuracy"] == "true"]
    qrte_hits = [row for row in rows if row["qrte_hit_chemical_accuracy"] == "true"]
    if len(qkud_hits) != int(args.expected_qkud_hits):
        raise SystemExit(f"Expected {args.expected_qkud_hits} QKUD hits, found {len(qkud_hits)}")
    if len(qrte_hits) != int(args.expected_qrte_hits):
        raise SystemExit(f"Expected {args.expected_qrte_hits} QRTE hits, found {len(qrte_hits)}")

    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    _write_rows(rows, csv_path)
    _plot(rows, png_path, pdf_path)

    qkud_only = [row["case_id"] for row in rows if row["qkud_only_hit"] == "true"]
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(
        f"Included {len(adapt_hits)} ADAPT hits, "
        f"{len(qkud_hits)} QKUD eps=0.05 hits, and {len(qrte_hits)} QRTE dt=0.05 hits."
    )
    print(f"Highlighted QKUD-only hits: {', '.join(qkud_only) if qkud_only else 'none'}")


if __name__ == "__main__":
    main()
