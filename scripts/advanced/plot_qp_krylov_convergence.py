from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from advanced.n2_fixed_budget_docs_regression import DOCS_RESULTS, load_golden_docs_row  # noqa: E402


DEFAULT_CASE = "n2_cas12_12_R2p50"
DEFAULT_METHOD = "ccik_thick_qp"
DEFAULT_PHASE = "kprop"
DEFAULT_BUDGET = 512
DEFAULT_FIXED_STORE_KEEP = 65_536
DEFAULT_RESTART_M_CYCLE = 30
DEFAULT_RESTART_NROOT = 8
DEFAULT_FIRST_CYCLE_SEED_COUNT = 1

LOG_LINE_RE = re.compile(
    r"ccik\(thick\)\s+cycle=\s*(?P<cycle>\d+)\s+iter=\s*(?P<iter>\d+)\s+"
    r"total=\s*(?P<total>[+-]?\d+\.\d+)\s+ref=\s*(?P<ref>[+-]?\d+\.\d+)\s+"
    r"delta_mEh=\s*(?P<delta>[+-]?\d+\.\d+)"
)


@dataclass(frozen=True)
class IterationPoint:
    cycle: int
    iter: int
    cum_new_krylov: int
    delta_mEh: float
    total_energy_eh: float
    reference_total_energy_eh: float


@dataclass(frozen=True)
class IterationTargetSelection:
    point: IterationPoint
    selection_mode: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot QP thick-restart energy convergence versus cumulative new Krylov vectors.")
    p.add_argument("--log", type=Path, required=True, help="Path to a live or completed ccik_stdout.log from a thick-restart QP run.")
    p.add_argument("--outdir", type=Path, default=None, help="Directory for CSV/JSON/plot outputs. Defaults to the log directory.")
    p.add_argument("--docs-results", type=Path, default=DOCS_RESULTS, help="Retained docs results.csv used for the published reference line.")
    p.add_argument("--system", default=DEFAULT_CASE)
    p.add_argument("--method", default=DEFAULT_METHOD)
    p.add_argument("--phase", default=DEFAULT_PHASE)
    p.add_argument("--budget-value", type=int, default=DEFAULT_BUDGET)
    p.add_argument("--fixed-store-keep", type=int, default=DEFAULT_FIXED_STORE_KEEP)
    p.add_argument("--restart-m-cycle", type=int, default=DEFAULT_RESTART_M_CYCLE)
    p.add_argument("--restart-nroot", type=int, default=DEFAULT_RESTART_NROOT)
    p.add_argument("--first-cycle-seed-count", type=int, default=DEFAULT_FIRST_CYCLE_SEED_COUNT)
    p.add_argument("--recent-window", type=int, default=80, help="Number of most recent iterations to show in the zoom panel.")
    p.add_argument("--watch", action="store_true", help="Refresh the plot while the log is still growing.")
    p.add_argument("--poll-s", type=float, default=5.0, help="Polling interval in seconds for --watch mode.")
    p.add_argument(
        "--max-idle-polls",
        type=int,
        default=2,
        help="Stop watch mode after this many unchanged polls once the log contains a final energy line.",
    )
    p.add_argument("--title", default=None, help="Optional plot title override.")
    return p.parse_args(argv)


def cumulative_new_krylov(
    cycle: int,
    iter_idx: int,
    *,
    restart_m_cycle: int,
    restart_nroot: int,
    first_cycle_seed_count: int = 1,
) -> int:
    if cycle < 1:
        raise ValueError(f"cycle must be >= 1, got {cycle}")
    if iter_idx < 1:
        raise ValueError(f"iter_idx must be >= 1, got {iter_idx}")
    if restart_m_cycle < 1:
        raise ValueError(f"restart_m_cycle must be >= 1, got {restart_m_cycle}")
    if restart_nroot < 0:
        raise ValueError(f"restart_nroot must be >= 0, got {restart_nroot}")
    if first_cycle_seed_count < 1:
        raise ValueError(f"first_cycle_seed_count must be >= 1, got {first_cycle_seed_count}")
    if cycle == 1:
        if iter_idx < first_cycle_seed_count:
            raise ValueError(
                f"iter_idx={iter_idx} is smaller than the first-cycle seed count={first_cycle_seed_count}"
            )
        return iter_idx - first_cycle_seed_count
    if iter_idx < restart_nroot:
        raise ValueError(f"iter_idx={iter_idx} is smaller than restart_nroot={restart_nroot} for cycle={cycle}")
    generated_in_cycle1 = max(0, restart_m_cycle - first_cycle_seed_count)
    generated_in_later_cycle = max(0, restart_m_cycle - restart_nroot)
    return generated_in_cycle1 + (cycle - 2) * generated_in_later_cycle + (iter_idx - restart_nroot)


def parse_thick_iteration_log(
    log_path: Path,
    *,
    restart_m_cycle: int,
    restart_nroot: int,
    first_cycle_seed_count: int = 1,
) -> list[IterationPoint]:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing thick-restart log: {log_path}")
    points: list[IterationPoint] = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        match = LOG_LINE_RE.search(raw)
        if match is None:
            continue
        cycle = int(match.group("cycle"))
        iter_idx = int(match.group("iter"))
        points.append(
            IterationPoint(
                cycle=cycle,
                iter=iter_idx,
                cum_new_krylov=cumulative_new_krylov(
                    cycle,
                    iter_idx,
                    restart_m_cycle=restart_m_cycle,
                    restart_nroot=restart_nroot,
                    first_cycle_seed_count=first_cycle_seed_count,
                ),
                delta_mEh=float(match.group("delta")),
                total_energy_eh=float(match.group("total")),
                reference_total_energy_eh=float(match.group("ref")),
            )
        )
    if not points:
        raise RuntimeError(f"No thick-restart iteration lines were parsed from {log_path}")
    return points


def first_iteration_beating_reference(points: list[IterationPoint], published_delta_mEh: float) -> IterationPoint | None:
    for point in points:
        if point.delta_mEh <= published_delta_mEh:
            return point
    return None


def select_point_for_cumulative_target(points: list[IterationPoint], target_cum_new_krylov: int) -> IterationTargetSelection:
    for point in points:
        if point.cum_new_krylov == target_cum_new_krylov:
            return IterationTargetSelection(point=point, selection_mode="exact_target")
    for point in points:
        if point.cum_new_krylov > target_cum_new_krylov:
            return IterationTargetSelection(point=point, selection_mode="first_after_target")
    return IterationTargetSelection(point=points[-1], selection_mode="latest_before_target")


def log_has_final_energy(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    return "CCIK_FINAL_TOTAL_ENERGY" in log_path.read_text(encoding="utf-8")


def _write_csv(points: list[IterationPoint], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=("cycle", "iter", "cum_new_krylov", "delta_mEh", "total_energy_eh", "reference_total_energy_eh"),
        )
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def _write_json(
    *,
    path: Path,
    log_path: Path,
    points: list[IterationPoint],
    published_row: dict[str, object],
    beat_point: IterationPoint | None,
    restart_m_cycle: int,
    restart_nroot: int,
    first_cycle_seed_count: int,
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path),
        "log_mtime_utc": datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "published_row": published_row,
        "restart_shape": {
            "restart_m_cycle": restart_m_cycle,
            "restart_nroot": restart_nroot,
            "first_cycle_seed_count": first_cycle_seed_count,
        },
        "num_points": len(points),
        "first_point": asdict(points[0]),
        "latest_point": asdict(points[-1]),
        "first_iteration_beating_published_row": None if beat_point is None else asdict(beat_point),
        "points": [asdict(point) for point in points],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot_convergence(
    *,
    points: list[IterationPoint],
    published_row: dict[str, object],
    beat_point: IterationPoint | None,
    output_png: Path,
    output_pdf: Path,
    title: str,
    recent_window: int,
) -> None:
    xvals = [point.cum_new_krylov for point in points]
    yvals = [point.delta_mEh for point in points]
    published_delta = float(published_row["delta_vs_ref_mEh"])
    recent_points = points[-max(1, int(recent_window)) :]
    x_recent = [point.cum_new_krylov for point in recent_points]
    y_recent = [point.delta_mEh for point in recent_points]

    fig, (ax, ax_recent) = plt.subplots(
        2,
        1,
        figsize=(10.25, 8.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.6, 1.4]},
    )
    ax.semilogy(
        xvals,
        yvals,
        color="#005f73",
        linewidth=1.8,
        marker="o",
        markersize=2.2,
        markerfacecolor="white",
        markeredgewidth=0.6,
        label="Live rerun",
    )
    ax.axhline(
        published_delta,
        color="#bb3e03",
        linestyle="--",
        linewidth=1.5,
        label=f"Published row = {published_delta:.6f} mEh",
    )

    if beat_point is not None:
        ax.scatter(
            [beat_point.cum_new_krylov],
            [beat_point.delta_mEh],
            color="#ae2012",
            s=40,
            zorder=5,
        )
        ax.annotate(
            (
                "First beat of published row\n"
                f"cycle={beat_point.cycle}, iter={beat_point.iter}, "
                f"cum_new={beat_point.cum_new_krylov}, "
                f"delta={beat_point.delta_mEh:.6f} mEh"
            ),
            xy=(beat_point.cum_new_krylov, beat_point.delta_mEh),
            xytext=(16, 14),
            textcoords="offset points",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#ae2012", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#ae2012", "lw": 1.0},
        )

    latest = points[-1]
    ax.annotate(
        (
            "Latest parsed point\n"
            f"cycle={latest.cycle}, iter={latest.iter}, "
            f"cum_new={latest.cum_new_krylov}, "
            f"delta={latest.delta_mEh:.6f} mEh"
        ),
        xy=(latest.cum_new_krylov, latest.delta_mEh),
        xytext=(-10, -52),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#0a9396", "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": "#0a9396", "lw": 1.0},
    )

    ax.text(
        0.99,
        0.98,
        (
            f"Published target: {published_row['system']} / {published_row['phase']} / "
            f"Kprop={published_row['fixed_propagate_keep']} / Kstore={published_row['fixed_store_keep']}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#6c757d", "alpha": 0.92},
    )

    ax.set_title(title)
    ax.set_xlabel("Cumulative New Krylov Vectors")
    ax.set_ylabel("delta_vs_ref_mEh")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(loc="upper right")

    ax_recent.plot(
        x_recent,
        y_recent,
        color="#0a9396",
        linewidth=1.4,
        marker="o",
        markersize=3.0,
        markerfacecolor="#94d2bd",
        markeredgewidth=0.6,
        markeredgecolor="#005f73",
    )
    ax_recent.axhline(
        published_delta,
        color="#bb3e03",
        linestyle="--",
        linewidth=1.2,
    )
    if beat_point is not None:
        ax_recent.scatter(
            [beat_point.cum_new_krylov],
            [beat_point.delta_mEh],
            color="#ae2012",
            s=38,
            zorder=5,
        )
    ax_recent.set_title(f"Recent {len(recent_points)} logged iterations")
    ax_recent.set_xlabel("Cumulative New Krylov Vectors")
    ax_recent.set_ylabel("delta_vs_ref_mEh")
    ax_recent.grid(True, alpha=0.28)

    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def _render_outputs(
    *,
    log_path: Path,
    outdir: Path,
    published_row: dict[str, object],
    restart_m_cycle: int,
    restart_nroot: int,
    first_cycle_seed_count: int,
    plot_title: str,
    recent_window: int,
) -> dict[str, object]:
    points = parse_thick_iteration_log(
        log_path,
        restart_m_cycle=restart_m_cycle,
        restart_nroot=restart_nroot,
        first_cycle_seed_count=first_cycle_seed_count,
    )
    beat_point = first_iteration_beating_reference(points, float(published_row["delta_vs_ref_mEh"]))

    csv_path = outdir / "krylov_convergence.csv"
    json_path = outdir / "krylov_convergence.json"
    png_path = outdir / "krylov_convergence.png"
    pdf_path = outdir / "krylov_convergence.pdf"

    _write_csv(points, csv_path)
    _write_json(
        path=json_path,
        log_path=log_path,
        points=points,
        published_row=published_row,
        beat_point=beat_point,
        restart_m_cycle=restart_m_cycle,
        restart_nroot=restart_nroot,
        first_cycle_seed_count=first_cycle_seed_count,
    )
    _plot_convergence(
        points=points,
        published_row=published_row,
        beat_point=beat_point,
        output_png=png_path,
        output_pdf=pdf_path,
        title=plot_title,
        recent_window=recent_window,
    )
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
        "num_points": len(points),
        "latest_point": asdict(points[-1]),
        "beat_point": None if beat_point is None else asdict(beat_point),
        "finalized": log_has_final_energy(log_path),
        "log_mtime": log_path.stat().st_mtime,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_path = args.log.resolve()
    outdir = log_path.parent if args.outdir is None else args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    published_row = load_golden_docs_row(
        path=args.docs_results,
        case_name=str(args.system),
        phase=str(args.phase),
        budget_value=int(args.budget_value),
        method=str(args.method),
    )
    if int(published_row["fixed_store_keep"]) != int(args.fixed_store_keep):
        raise ValueError(
            f"Published row fixed_store_keep={published_row['fixed_store_keep']} does not match "
            f"--fixed-store-keep={args.fixed_store_keep}"
        )
    plot_title = args.title
    if plot_title is None:
        plot_title = (
            f"{published_row['title']} | {published_row['method']} | {published_row['phase']} | "
            f"Kprop={published_row['fixed_propagate_keep']} Kstore={published_row['fixed_store_keep']}"
        )
    if not args.watch:
        result = _render_outputs(
            log_path=log_path,
            outdir=outdir,
            published_row=published_row,
            restart_m_cycle=int(args.restart_m_cycle),
            restart_nroot=int(args.restart_nroot),
            first_cycle_seed_count=int(args.first_cycle_seed_count),
            plot_title=plot_title,
            recent_window=int(args.recent_window),
        )
        print(json.dumps({k: result[k] for k in ("csv", "json", "png", "pdf", "num_points", "latest_point", "beat_point")}, indent=2))
        return 0

    last_mtime: float | None = None
    idle_polls = 0
    while True:
        if not log_path.exists():
            time.sleep(max(0.1, float(args.poll_s)))
            continue
        current_mtime = log_path.stat().st_mtime
        if last_mtime is None or current_mtime != last_mtime:
            result = _render_outputs(
                log_path=log_path,
                outdir=outdir,
                published_row=published_row,
                restart_m_cycle=int(args.restart_m_cycle),
                restart_nroot=int(args.restart_nroot),
                first_cycle_seed_count=int(args.first_cycle_seed_count),
                plot_title=plot_title,
                recent_window=int(args.recent_window),
            )
            last_mtime = current_mtime
            idle_polls = 0
            latest = result["latest_point"]
            print(
                json.dumps(
                    {
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "num_points": result["num_points"],
                        "latest_point": latest,
                        "beat_point": result["beat_point"],
                        "finalized": result["finalized"],
                    },
                    indent=2,
                ),
                flush=True,
            )
            if result["finalized"]:
                idle_polls += 1
        else:
            if log_has_final_energy(log_path):
                idle_polls += 1
            if idle_polls >= int(args.max_idle_polls):
                break
        time.sleep(max(0.1, float(args.poll_s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
