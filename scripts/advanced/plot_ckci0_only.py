#!/usr/bin/env python
"""Plot CKCI-0/exactized traces without the underlying CK curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHEMICAL_ACCURACY_MHA = 1.593601


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_label(run: dict[str, Any]) -> str:
    params = dict(run.get("solver", {}).get("params", {}))
    kstore = params.get("fixed_store_keep")
    kprop = params.get("fixed_propagate_keep")
    pieces = []
    if kstore is not None:
        pieces.append(f"Kstore={int(kstore):,}")
    if kprop is not None:
        pieces.append(f"Kprop={int(kprop):,}")
    return "CKCI-0 " + " ".join(pieces) if pieces else f"CKCI-0 {run.get('label', run.get('name', 'run'))}"


def _points(rows: list[dict[str, str]], *, max_iter: int | None) -> tuple[list[int], list[float], list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    neg_xs: list[int] = []
    neg_ys: list[float] = []
    for row in rows:
        raw_y = row.get("exactized_abs_delta_mHa")
        raw_x = row.get("cum_new_krylov")
        if raw_y in (None, "", "None") or raw_x in (None, "", "None"):
            continue
        x = int(float(raw_x))
        if max_iter is not None and x > max_iter:
            continue
        y = max(1.0e-12, float(raw_y))
        xs.append(x)
        ys.append(y)
        raw_delta = row.get("exactized_delta_mHa")
        if raw_delta not in (None, "", "None") and float(raw_delta) < 0.0:
            neg_xs.append(x)
            neg_ys.append(y)
    return xs, ys, neg_xs, neg_ys


def plot_ckci0(case_dir: Path, *, output_stem: str = "energy_error_vs_iteration_ckci0_comparison") -> dict[str, Any]:
    input_path = case_dir / "input.json"
    spec = json.loads(input_path.read_text(encoding="utf-8"))
    max_iter = spec.get("study", {}).get("plot_max_cumulative_iterations", spec.get("study", {}).get("max_cumulative_iterations"))
    max_iter = None if max_iter is None else int(max_iter)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    plotted = 0
    summary_runs: dict[str, Any] = {}
    for run in spec.get("runs", []):
        run_name = str(run["name"])
        trace_path = case_dir / run_name / "iteration_trace.csv"
        if not trace_path.exists():
            summary_runs[run_name] = {"trace_path": str(trace_path), "points": 0, "status": "missing_trace"}
            continue
        xs, ys, neg_xs, neg_ys = _points(_read_rows(trace_path), max_iter=max_iter)
        if not xs:
            summary_runs[run_name] = {"trace_path": str(trace_path), "points": 0, "status": "missing_ckci0_points"}
            continue
        ax.plot(xs, ys, marker="o", markersize=3.3, linewidth=1.45, label=_run_label(run))
        if neg_xs:
            ax.scatter(neg_xs, neg_ys, marker="x", s=42, color="#B3261E", linewidths=1.5, zorder=5)
        hit = next((x for x, y in zip(xs, ys, strict=False) if y <= CHEMICAL_ACCURACY_MHA), None)
        summary_runs[run_name] = {
            "trace_path": str(trace_path),
            "points": len(xs),
            "first_iteration_below_1kcal_mol": hit,
            "best_abs_delta_mHa": min(ys),
            "final_abs_delta_mHa": ys[-1],
            "status": "ok",
        }
        plotted += 1

    if plotted == 0:
        raise RuntimeError(f"No populated CKCI-0/exactized rows found under {case_dir}")

    ax.axhline(
        CHEMICAL_ACCURACY_MHA,
        color="#B3261E",
        linestyle="--",
        linewidth=1.15,
        alpha=0.82,
        label="Chemical accuracy (1 kcal/mol)",
    )
    ax.set_xlabel("Cumulative Krylov iterations")
    ax.set_ylabel("Absolute energy error (mHa)")
    ax.set_yscale(str(spec.get("study", {}).get("plot_yscale", "log")))
    if max_iter is not None:
        ax.set_xlim(0, max_iter)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    ax.set_title(str(spec.get("label", case_dir.name)) + " | CKCI-0")
    ax.legend(fontsize=8.0, frameon=False, ncol=1)

    png_path = case_dir / f"{output_stem}.png"
    pdf_path = case_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "case_dir": str(case_dir),
        "input_json": str(input_path),
        "plot_png": str(png_path),
        "plot_pdf": str(pdf_path),
        "chemical_accuracy_mHa": CHEMICAL_ACCURACY_MHA,
        "plotted_runs": plotted,
        "runs": summary_runs,
    }
    (case_dir / f"{output_stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CKCI-0-only comparison from exactized trace columns")
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output-stem", default="energy_error_vs_iteration_ckci0_comparison")
    args = parser.parse_args()
    summary = plot_ckci0(args.case_dir, output_stem=args.output_stem)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
