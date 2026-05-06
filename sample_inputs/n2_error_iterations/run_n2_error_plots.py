#!/usr/bin/env python3
"""Run the tracked N2 teaching samples and build CCIK/CKCI-0 plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SAMPLE_ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
ADVANCED_SCRIPTS = SCRIPTS / "advanced"
SRC = ROOT / "src"
for path in (str(ROOT), str(SCRIPTS), str(ADVANCED_SCRIPTS), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import n2_thick_iteration_study as n2_study  # noqa: E402


DEFAULT_INPUTS = (
    SAMPLE_ROOT / "cases" / "n2_cas10_10_k1000_s050000" / "input.json",
    SAMPLE_ROOT / "cases" / "n2_cas12_12_k1000_s050000" / "input.json",
)
DEFAULT_PLOT_STEM = "n2_error_vs_iteration_k1000_s050000"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_name(spec: dict[str, Any]) -> str:
    runs = spec.get("runs", [])
    if len(runs) != 1:
        raise ValueError(f"Sample input must contain exactly one run: {spec.get('name')}")
    return str(runs[0]["name"])


def _load_trace(input_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = _load_json(input_path)
    trace_path = input_path.parent / _run_name(spec) / "iteration_trace.json"
    if not trace_path.exists():
        raise FileNotFoundError(
            f"Missing trace file {trace_path}. Run without --skip-run first, "
            "or re-run with --force if old outputs are stale."
        )
    rows = _load_json(trace_path)
    if not rows:
        raise RuntimeError(f"Trace file is empty: {trace_path}")
    return spec, rows


def _positive_error(value: Any) -> float:
    return max(1.0e-12, float(value))


def _plot_combined(
    *,
    traces: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    output_dir: Path,
    stem: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    palette = ("#0B5FFF", "#E67E22", "#1B9E77", "#8E44AD")
    max_iter = 0
    ckci_count = 0

    for idx, (spec, rows) in enumerate(traces):
        color = palette[idx % len(palette)]
        label = str(spec["label"])
        xvals = [int(row["cum_new_krylov"]) for row in rows]
        yvals = [_positive_error(row["abs_delta_mHa"]) for row in rows]
        max_iter = max(max_iter, max(xvals))
        ax.plot(
            xvals,
            yvals,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=2.7,
            label=f"{label} | CCIK",
        )

        ckci_points = [
            (int(row["cum_new_krylov"]), _positive_error(row["exactized_abs_delta_mHa"]))
            for row in rows
            if row.get("exactized_abs_delta_mHa") is not None
        ]
        ckci_count += len(ckci_points)
        if ckci_points:
            ax.plot(
                [point[0] for point in ckci_points],
                [point[1] for point in ckci_points],
                color=color,
                linestyle="--",
                linewidth=1.6,
                marker="s",
                markersize=2.7,
                label=f"{label} | CKCI-0",
            )

    ax.set_yscale("log")
    ax.set_xlim(0, max(300, max_iter))
    ax.set_xlabel("Cumulative new Krylov vectors")
    ax.set_ylabel("|Energy error| (mHa)")
    ax.set_title("N2 error vs iteration | Kprop=1000 Kstore=50000")
    ax.grid(True, which="both", alpha=0.25)
    ax.axhline(1.0, color="#C0392B", linestyle=":", linewidth=1.2, label="1 mHa")
    ax.legend(fontsize=8)
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)

    if ckci_count == 0:
        print("warning: no CKCI-0 checkpoint rows were found in the traces", file=sys.stderr)
    return png_path, pdf_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        dest="input_files",
        help="Sample input.json to run. Defaults to the two tracked N2 samples.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate case outputs even if study_summary.json exists.")
    parser.add_argument("--skip-run", action="store_true", help="Only rebuild the combined plot from existing traces.")
    parser.add_argument("--output-dir", type=Path, default=SAMPLE_ROOT / "plots")
    parser.add_argument("--plot-stem", default=DEFAULT_PLOT_STEM)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_files = tuple(args.input_files or DEFAULT_INPUTS)
    missing = [path for path in input_files if not path.exists()]
    if missing:
        raise SystemExit("Missing sample input(s):\n" + "\n".join(str(path) for path in missing))

    summaries = []
    if not args.skip_run:
        for input_path in input_files:
            summaries.append(n2_study.run_case(input_path, force=bool(args.force)))

    traces = [_load_trace(input_path) for input_path in input_files]
    png_path, pdf_path = _plot_combined(traces=traces, output_dir=args.output_dir, stem=str(args.plot_stem))
    print(
        json.dumps(
            {
                "input_files": [str(path) for path in input_files],
                "case_summaries": summaries,
                "combined_plot_png": str(png_path),
                "combined_plot_pdf": str(pdf_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
