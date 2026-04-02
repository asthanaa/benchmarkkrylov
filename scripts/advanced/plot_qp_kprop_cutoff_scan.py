from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from advanced.n2_fixed_budget_docs_regression import DOCS_RESULTS, _load_docs_rows, load_golden_docs_row  # noqa: E402
from advanced.plot_qp_krylov_convergence import (  # noqa: E402
    DEFAULT_CASE,
    DEFAULT_FIXED_STORE_KEEP,
    DEFAULT_FIRST_CYCLE_SEED_COUNT,
    DEFAULT_METHOD,
    DEFAULT_PHASE,
    DEFAULT_RESTART_M_CYCLE,
    DEFAULT_RESTART_NROOT,
    parse_thick_iteration_log,
    select_point_for_cumulative_target,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a staged kprop scan plot using the energy after a fixed cumulative new-Krylov count.")
    p.add_argument("--stage-root", type=Path, required=True, help="Staged publication output folder containing runs/ and results.csv.")
    p.add_argument("--docs-results", type=Path, default=DOCS_RESULTS)
    p.add_argument("--system", default=DEFAULT_CASE)
    p.add_argument("--method", default=DEFAULT_METHOD)
    p.add_argument("--phase", default=DEFAULT_PHASE)
    p.add_argument("--fixed-store-keep", type=int, default=DEFAULT_FIXED_STORE_KEEP)
    p.add_argument("--target-cum-new-krylov", type=int, default=100)
    p.add_argument("--restart-m-cycle", type=int, default=DEFAULT_RESTART_M_CYCLE)
    p.add_argument("--restart-nroot", type=int, default=DEFAULT_RESTART_NROOT)
    p.add_argument("--first-cycle-seed-count", type=int, default=DEFAULT_FIRST_CYCLE_SEED_COUNT)
    p.add_argument("--title", default=None)
    return p.parse_args(argv)


def _published_budget_rows(
    *,
    docs_results: Path,
    system: str,
    method: str,
    phase: str,
    fixed_store_keep: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in _load_docs_rows(docs_results):
        if raw["system"] != system:
            continue
        if raw["method"] != method:
            continue
        if raw["phase"] != phase:
            continue
        if int(raw["fixed_store_keep"]) != int(fixed_store_keep):
            continue
        rows.append(
            {
                "budget_value": int(raw["budget_value"]),
                "fixed_propagate_keep": int(raw["fixed_propagate_keep"]),
                "fixed_store_keep": int(raw["fixed_store_keep"]),
                "e_total_eh": float(raw["e_total_eh"]),
                "delta_vs_ref_mEh": float(raw["delta_vs_ref_mEh"]),
                "reference_total_energy_eh": float(raw["reference_total_energy_eh"]),
            }
        )
    rows.sort(key=lambda row: int(row["budget_value"]))
    return rows


def _discover_kprop_logs(stage_root: Path, *, system: str, method: str, phase: str) -> list[tuple[int, Path]]:
    base = stage_root / "runs" / system / method / phase
    if not base.exists():
        return []
    found: list[tuple[int, Path]] = []
    for child in sorted(base.iterdir(), key=lambda path: int(path.name)):
        if not child.is_dir():
            continue
        log_path = child / "ccik_stdout.log"
        if log_path.exists():
            found.append((int(child.name), log_path))
    return found


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "budget_value",
                "cum_new_krylov_target",
                "selection_mode",
                "cycle",
                "iter",
                "cum_new_krylov",
                "delta_mEh",
                "total_energy_eh",
                "reference_total_energy_eh",
                "published_final_delta_mEh",
                "published_final_total_energy_eh",
                "log_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(
    *,
    path: Path,
    stage_root: Path,
    target_cum_new_krylov: int,
    rows: list[dict[str, object]],
    published_rows: list[dict[str, object]],
    missing_budgets: list[int],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage_root": str(stage_root),
        "target_cum_new_krylov": int(target_cum_new_krylov),
        "num_rows": len(rows),
        "rows": rows,
        "published_rows": published_rows,
        "missing_budgets": missing_budgets,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot(
    *,
    rows: list[dict[str, object]],
    published_rows: list[dict[str, object]],
    output_png: Path,
    output_pdf: Path,
    title: str,
    target_cum_new_krylov: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.75), constrained_layout=True)
    if published_rows:
        ax.semilogy(
            [int(row["budget_value"]) for row in published_rows],
            [float(row["delta_vs_ref_mEh"]) for row in published_rows],
            linestyle="--",
            marker="s",
            markersize=5.0,
            color="#bb3e03",
            linewidth=1.5,
            label="Published final",
        )
    if rows:
        exact = [row for row in rows if row["selection_mode"] == "exact_target"]
        approx = [row for row in rows if row["selection_mode"] != "exact_target"]
        if exact:
            ax.semilogy(
                [int(row["budget_value"]) for row in exact],
                [float(row["delta_mEh"]) for row in exact],
                marker="o",
                markersize=6.0,
                linewidth=1.8,
                color="#005f73",
                label=f"Live at cum_new={target_cum_new_krylov}",
            )
        if approx:
            ax.semilogy(
                [int(row["budget_value"]) for row in approx],
                [float(row["delta_mEh"]) for row in approx],
                marker="D",
                markersize=6.0,
                linewidth=1.5,
                color="#ca6702",
                label=f"Live nearest to cum_new={target_cum_new_krylov}",
            )
        for row in rows:
            ax.annotate(
                f"{row['delta_mEh']:.3f} mEh\nc{row['cycle']} i{row['iter']}",
                xy=(int(row["budget_value"]), float(row["delta_mEh"])),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fixed_propagate_keep")
    ax.set_ylabel("delta_vs_ref_mEh")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="best")
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def build_cutoff_rows(
    *,
    stage_root: Path,
    docs_results: Path,
    system: str,
    method: str,
    phase: str,
    fixed_store_keep: int,
    target_cum_new_krylov: int,
    restart_m_cycle: int,
    restart_nroot: int,
    first_cycle_seed_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    published_rows = _published_budget_rows(
        docs_results=docs_results,
        system=system,
        method=method,
        phase=phase,
        fixed_store_keep=fixed_store_keep,
    )
    published_by_budget = {int(row["budget_value"]): row for row in published_rows}
    rows: list[dict[str, object]] = []
    for budget_value, log_path in _discover_kprop_logs(stage_root, system=system, method=method, phase=phase):
        points = parse_thick_iteration_log(
            log_path,
            restart_m_cycle=restart_m_cycle,
            restart_nroot=restart_nroot,
            first_cycle_seed_count=first_cycle_seed_count,
        )
        selection = select_point_for_cumulative_target(points, target_cum_new_krylov)
        published = published_by_budget.get(budget_value)
        if published is None:
            published = load_golden_docs_row(
                path=docs_results,
                case_name=system,
                phase=phase,
                budget_value=budget_value,
                method=method,
            )
        rows.append(
            {
                "budget_value": int(budget_value),
                "cum_new_krylov_target": int(target_cum_new_krylov),
                "selection_mode": selection.selection_mode,
                "cycle": int(selection.point.cycle),
                "iter": int(selection.point.iter),
                "cum_new_krylov": int(selection.point.cum_new_krylov),
                "delta_mEh": float(selection.point.delta_mEh),
                "total_energy_eh": float(selection.point.total_energy_eh),
                "reference_total_energy_eh": float(selection.point.reference_total_energy_eh),
                "published_final_delta_mEh": float(published["delta_vs_ref_mEh"]),
                "published_final_total_energy_eh": float(published["e_total_eh"]),
                "log_path": str(log_path),
            }
        )
    rows.sort(key=lambda row: int(row["budget_value"]))
    missing_budgets = sorted(set(int(row["budget_value"]) for row in published_rows) - set(int(row["budget_value"]) for row in rows))
    return rows, published_rows, missing_budgets


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stage_root = args.stage_root.resolve()
    rows, published_rows, missing_budgets = build_cutoff_rows(
        stage_root=stage_root,
        docs_results=args.docs_results,
        system=str(args.system),
        method=str(args.method),
        phase=str(args.phase),
        fixed_store_keep=int(args.fixed_store_keep),
        target_cum_new_krylov=int(args.target_cum_new_krylov),
        restart_m_cycle=int(args.restart_m_cycle),
        restart_nroot=int(args.restart_nroot),
        first_cycle_seed_count=int(args.first_cycle_seed_count),
    )
    stem = f"{args.phase}_scan_cum{int(args.target_cum_new_krylov)}"
    csv_path = stage_root / f"{stem}.csv"
    json_path = stage_root / f"{stem}.json"
    png_path = stage_root / f"{stem}.png"
    pdf_path = stage_root / f"{stem}.pdf"
    _write_csv(rows, csv_path)
    _write_json(
        path=json_path,
        stage_root=stage_root,
        target_cum_new_krylov=int(args.target_cum_new_krylov),
        rows=rows,
        published_rows=published_rows,
        missing_budgets=missing_budgets,
    )
    title = args.title
    if title is None:
        title = (
            f"{args.system} | {args.method} | {args.phase} | "
            f"energy after cum_new_krylov={int(args.target_cum_new_krylov)}"
        )
    _plot(
        rows=rows,
        published_rows=published_rows,
        output_png=png_path,
        output_pdf=pdf_path,
        title=title,
        target_cum_new_krylov=int(args.target_cum_new_krylov),
    )
    print(
        json.dumps(
            {
                "csv": str(csv_path),
                "json": str(json_path),
                "png": str(png_path),
                "pdf": str(pdf_path),
                "num_rows": len(rows),
                "missing_budgets": missing_budgets,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
