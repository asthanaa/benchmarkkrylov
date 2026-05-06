#!/usr/bin/env python3
"""Make a combined CK trajectory and CKCI sidecar plot for a finished study.

The CK trajectory is plotted against cumulative new Krylov vectors. The CKCI
sidecar is plotted against determinant count, because CKCI stages are
postprocess determinant-space refinements rather than additional Krylov steps.
Points below the chosen reference are marked with a red cross.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import n2_thick_iteration_study as n2_study  # noqa: E402


CHEMICAL_ACCURACY_MHA = 1.593601438


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_ck_rows(case_dir: Path, run_name: str) -> list[dict[str, Any]]:
    _, trace_json = n2_study._trace_paths(case_dir / run_name)
    rows = n2_study.load_trace_rows_from_json(trace_json)
    if not rows:
        raise FileNotFoundError(f"No CK iteration rows found at {trace_json}")
    return rows


def _load_ckci_rows(case_dir: Path) -> list[dict[str, Any]]:
    summary_path = case_dir / "ckci_n" / "ckci_n_summary.json"
    if not summary_path.exists():
        return []
    payload = _load_json(summary_path)
    return [dict(row) for row in payload.get("stages", [])]


def _positive_abs_mha(row: dict[str, Any]) -> float:
    return max(1.0e-12, float(row["abs_delta_mHa"]))


def _below_reference(row: dict[str, Any]) -> bool:
    delta = row.get("delta_mHa")
    return delta is not None and float(delta) < -1.0e-9


def _clean_label(text: str, max_len: int = 95) -> str:
    stripped = " ".join(str(text).replace("|", " | ").split())
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 3].rstrip() + "..."


def plot_combined(
    *,
    case_dir: Path,
    run_name: str,
    output_prefix: Path,
    max_cumulative_iterations: int | None = None,
) -> tuple[Path, Path]:
    case_dir = case_dir.resolve()
    spec = _load_json(case_dir / "input.json")
    reference_choice_path = case_dir / "system_setup" / "reference_choice.json"
    reference_choice = _load_json(reference_choice_path) if reference_choice_path.exists() else {}
    ck_rows = _load_ck_rows(case_dir, run_name)
    ckci_rows = _load_ckci_rows(case_dir)

    if max_cumulative_iterations is None:
        max_cumulative_iterations = int(
            spec.get("study", {}).get(
                "plot_max_cumulative_iterations",
                spec.get("study", {}).get("max_cumulative_iterations", ck_rows[-1]["cum_new_krylov"]),
            )
        )

    title = _clean_label(spec.get("label", case_dir.name))
    reference_label = str(reference_choice.get("chosen_reference_label") or reference_choice.get("reference_kind") or "reference")

    fig, (ax_ck, ax_ckci) = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.5),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    x_ck = [int(row["cum_new_krylov"]) for row in ck_rows]
    y_ck = [_positive_abs_mha(row) for row in ck_rows]
    ax_ck.plot(
        x_ck,
        y_ck,
        color="#174A7C",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
        label="CK trajectory",
    )
    below_ck = [(int(row["cum_new_krylov"]), _positive_abs_mha(row)) for row in ck_rows if _below_reference(row)]
    if below_ck:
        ax_ck.scatter(
            [point[0] for point in below_ck],
            [point[1] for point in below_ck],
            color="#B91C1C",
            marker="x",
            s=54,
            linewidths=1.5,
            zorder=6,
            label="Below reference",
        )
    ax_ck.axhline(
        CHEMICAL_ACCURACY_MHA,
        color="#6B7280",
        linestyle="--",
        linewidth=1.0,
        label="Chemical accuracy (1 kcal/mol)",
    )
    ax_ck.set_yscale("log")
    ax_ck.set_xlim(0, int(max_cumulative_iterations))
    ax_ck.set_xlabel("Cumulative new Krylov vectors")
    ax_ck.set_ylabel("|Energy error| (mHa)")
    ax_ck.set_title("CK convergence")
    ax_ck.grid(True, which="both", alpha=0.25)
    ax_ck.legend(frameon=False, fontsize=8)

    if ckci_rows:
        x_ci = [max(1, int(row["determinant_count"])) for row in ckci_rows]
        y_ci = [_positive_abs_mha(row) for row in ckci_rows]
        ax_ckci.plot(x_ci, y_ci, color="#4B5563", linewidth=1.5, alpha=0.8)
        colors = ["#111827", "#D97706", "#2563EB", "#059669", "#7C3AED"]
        markers = ["o", "s", "D", "^", "v"]
        for idx, row in enumerate(ckci_rows):
            marker = "x" if _below_reference(row) else markers[idx % len(markers)]
            color = "#B91C1C" if _below_reference(row) else colors[idx % len(colors)]
            ax_ckci.scatter(
                [x_ci[idx]],
                [y_ci[idx]],
                color=color,
                marker=marker,
                s=72,
                linewidths=1.5 if marker == "x" else 0.8,
                edgecolors="none" if marker == "x" else "white",
                zorder=5,
                label=str(row["stage_label"]),
            )
            ax_ckci.annotate(
                str(row["stage_label"]),
                (x_ci[idx], y_ci[idx]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
        ax_ckci.axhline(CHEMICAL_ACCURACY_MHA, color="#6B7280", linestyle="--", linewidth=1.0)
        ax_ckci.set_xscale("log")
        ax_ckci.set_yscale("log")
        ax_ckci.set_xlabel("Determinants")
        ax_ckci.set_title("CKCI postprocess")
        ax_ckci.grid(True, which="both", alpha=0.25)
    else:
        ax_ckci.text(
            0.5,
            0.5,
            "CKCI sidecar not available yet",
            ha="center",
            va="center",
            transform=ax_ckci.transAxes,
            color="#6B7280",
        )
        ax_ckci.set_axis_off()

    fig.suptitle(f"{title}\nReference: {reference_label}", fontsize=10)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot CK and CKCI results for one study case")
    parser.add_argument("--case", type=Path, required=True, help="Case directory containing input.json")
    parser.add_argument("--run", type=str, required=True, help="Run directory name")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help="Output path without suffix; defaults to CASE/ck_ckci_comparison",
    )
    parser.add_argument("--max-cumulative-iterations", type=int, default=None)
    args = parser.parse_args(argv)

    output_prefix = args.output_prefix
    if output_prefix is None:
        output_prefix = args.case / "ck_ckci_comparison"
    png_path, pdf_path = plot_combined(
        case_dir=args.case,
        run_name=args.run,
        output_prefix=output_prefix,
        max_cumulative_iterations=args.max_cumulative_iterations,
    )
    print(json.dumps({"plot_png": str(png_path), "plot_pdf": str(pdf_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
