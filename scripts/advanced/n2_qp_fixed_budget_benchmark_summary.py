"""Build a compact summary of the settled N2 fixed-budget QP benchmarks.

This script does not rerun chemistry. It reads the completed CSV artifacts for
the N2 CAS(12,12) and CAS(14,14) QP thick fixed-budget studies, copies the
finished sweep plots into the docs tree, and writes a consolidated figure that
shows the store/propagate trends side by side.
"""

from __future__ import annotations

from pathlib import Path
import csv
import shutil

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DOCS_FIG_DIR = ROOT / "docs" / "assets" / "figures"
CAS12_DIR = (
    ROOT
    / "experiments"
    / "threshold_test"
    / "n2_qp_n2_cas12_12_R2p50_fixed_budget_centered_20260326_live"
)
CAS14_DIR = (
    ROOT
    / "experiments"
    / "threshold_test"
    / "n2_qp_n2_cas14_14_R2p50_fixed_budget_centered_20260326_thick_prop2048_live"
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _thick_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["method"] == "thick" and not row["failure"]]


def _sorted_points(
    rows: list[dict[str, str]],
    *,
    sweep_kind: str,
    fixed_store_keep: int | None = None,
    fixed_propagate_keep: int | None = None,
) -> tuple[list[int], list[float]]:
    filtered = [row for row in rows if row["sweep_kind"] == sweep_kind]
    if fixed_store_keep is not None:
        filtered = [row for row in filtered if int(row["fixed_store_keep"]) == fixed_store_keep]
    if fixed_propagate_keep is not None:
        filtered = [row for row in filtered if int(row["fixed_propagate_keep"]) == fixed_propagate_keep]
    filtered.sort(key=lambda row: int(row["budget_value"]))
    x_values = [int(row["budget_value"]) for row in filtered]
    y_values = [float(row["abs_delta_vs_ref_mEh"]) for row in filtered]
    return x_values, y_values


def _copy_plot(source_dir: Path, src_name: str, dst_name: str) -> None:
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / src_name, DOCS_FIG_DIR / dst_name)


def build_summary_figure() -> Path:
    cas12_rows = _thick_rows(_read_rows(CAS12_DIR / "fixed_budget_sweep.csv"))
    cas14_rows = _thick_rows(_read_rows(CAS14_DIR / "fixed_budget_sweep.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    plot_specs = [
        (
            axes[0, 0],
            cas12_rows,
            "fixed_store_keep",
            {"fixed_propagate_keep": 512},
            "N2 CAS(12,12): store sweep at Kprop = 512",
            "fixed_store_keep",
        ),
        (
            axes[0, 1],
            cas12_rows,
            "fixed_propagate_keep",
            {"fixed_store_keep": 65536},
            "N2 CAS(12,12): propagate sweep at Kstore = 65536",
            "fixed_propagate_keep",
        ),
        (
            axes[1, 0],
            cas14_rows,
            "fixed_store_keep",
            {"fixed_propagate_keep": 2048},
            "N2 CAS(14,14): store sweep at Kprop = 2048",
            "fixed_store_keep",
        ),
        (
            axes[1, 1],
            cas14_rows,
            "fixed_propagate_keep",
            {"fixed_store_keep": 65536},
            "N2 CAS(14,14): propagate sweep at Kstore = 65536",
            "fixed_propagate_keep",
        ),
    ]

    for ax, rows, sweep_kind, filters, title, xlabel in plot_specs:
        x_values, y_values = _sorted_points(rows, sweep_kind=sweep_kind, **filters)
        ax.plot(x_values, y_values, marker="o", linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|\Delta E|$ vs reference / mEh")
        ax.grid(True, which="both", alpha=0.3)

        if y_values:
            best_idx = min(range(len(y_values)), key=y_values.__getitem__)
            ax.scatter([x_values[best_idx]], [y_values[best_idx]], color="crimson", zorder=5)
            ax.annotate(
                f"best = {y_values[best_idx]:.3g} mEh",
                xy=(x_values[best_idx], y_values[best_idx]),
                xytext=(8, 8),
                textcoords="offset points",
                color="crimson",
            )

    fig.suptitle(
        "Settled QP thick fixed-budget strategy on N2: tune propagate first, keep store moderate",
        fontsize=15,
    )

    output_path = DOCS_FIG_DIR / "n2_fixed_budget_strategy_summary.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    _copy_plot(
        CAS12_DIR,
        "fixed_store_keep_sweep.png",
        "n2_cas12_fixed_store_keep_sweep.png",
    )
    _copy_plot(
        CAS12_DIR,
        "fixed_propagate_keep_sweep.png",
        "n2_cas12_fixed_propagate_keep_sweep.png",
    )
    _copy_plot(
        CAS14_DIR,
        "fixed_store_keep_sweep.png",
        "n2_cas14_fixed_store_keep_sweep.png",
    )
    _copy_plot(
        CAS14_DIR,
        "fixed_propagate_keep_sweep.png",
        "n2_cas14_fixed_propagate_keep_sweep.png",
    )
    output_path = build_summary_figure()
    print(output_path)


if __name__ == "__main__":
    main()
