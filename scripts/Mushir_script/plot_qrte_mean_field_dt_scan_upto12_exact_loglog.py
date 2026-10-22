#!/usr/bin/env python
"""Create individual and combined log-log figures from validated exact QRTE scans."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchmarkkrylov-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, StrMethodFormatter
import numpy as np

import collect_qrte_mean_field_active_dt_scan4 as collector
from collect_qrte_mean_field_active_dt_scan_upto12_exact import LABELS
import run_qrte_mean_field_active_dt_scan_upto12_exact as scan

XLABEL = r"$\Delta t/(\pi/W_{\mathrm{MF}})$"
YLABEL = r"$N_{\mathrm{QKrylov}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)"
XTICKS = (.01, .025, .05, .1, .25, .5, 1., 2.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return scan.qrte.resolve_path(path).resolve().relative_to(scan.qrte.ROOT).as_posix()


def validated_rows(root, *, scan_module=scan):
    plan_path = root / "scan_plan.json"
    plan = scan.qrte.load_json(plan_path)
    args = scan_module.parse_args(["--outdir", str(root), "--n-steps", str(plan["n_steps"]),
                           "--overlap-tol", str(plan["overlap_tol"]),
                           "--max-workers", str(plan["max_workers"])])
    tasks = scan_module.load_tasks(args)
    if (plan["tasks"] != collector.plan_rows(tasks) or plan["factors"] != list(scan_module.FACTORS)
            or plan["case_ids"] != list(scan_module.CASE_IDS)
            or plan.get("evolution_mode", "trotter") != args.evolution_mode
            or plan["trotter_order"] != args.trotter_order or plan["trotter_steps"] != args.trotter_steps):
        raise ValueError("Saved plan differs from the requested scan")
    rows = collector.result_rows(tasks, args)
    expected = len(scan_module.CASE_IDS) * len(scan_module.FACTORS)
    if len(rows) != expected or any(r["status"] not in ("completed_hit", "completed_no_hit") for r in rows):
        raise ValueError(f"Plot requires all {expected} completed, validated conditions: {Counter(r['status'] for r in rows)}")
    standard_rows = [dict(row) for row in rows]
    sources = {plan_path}
    for task, row in zip(tasks, rows):
        run_dir = Path(row["summary_path"]).parent
        config_path, history_path = run_dir / "run_config.json", run_dir / "min_energy_history.csv"
        config = scan.qrte.load_json(config_path)
        if ((config["evolution_mode"], config["trotter_order"], config["trotter_steps"]) !=
                (args.evolution_mode, args.trotter_order, args.trotter_steps)
                or not math.isclose(config["delta_t"], task["delta_t"], rel_tol=1e-12)):
            raise ValueError(f"Config differs from planned propagation/timestep: {run_dir}")
        if digest(config["hamiltonian_cache_path"]) != task["case"]["hamiltonian_sha256"]:
            raise ValueError(f"Config selects another Hamiltonian: {run_dir}")
        with history_path.open(newline="") as handle:
            history = list(csv.DictReader(handle))
        if [int(p["iteration"]) for p in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Nonsequential history: {run_dir}")
        no_hit = row["status"] == "completed_no_hit"
        row.update(summary_path=relative(row["summary_path"]), history_path=relative(history_path),
                   config_path=relative(config_path), qubits=2 * task["case"]["active_orbitals"],
                   hamiltonian_cache_path=relative(task["case"]["hamiltonian_cache_path"]),
                   mean_field_reference_path=relative(task["case"]["mean_field_reference_path"]),
                   plotted_y=row["n_steps_completed"] if no_hit else row["hit_iteration"],
                   plotted_as_lower_bound=no_hit)
        sources.update((run_dir / "summary.json", config_path, history_path, run_dir / "input_snapshot.json",
                        Path(task["case"]["hamiltonian_cache_path"]),
                        Path(task["case"]["mean_field_reference_path"])))
        if (run_dir / "reused_from.json").exists():
            sources.add(run_dir / "reused_from.json")
    return rows, args, sources, standard_rows


def draw_series(ax, points, *, color="#0072B2", marker="o", linestyle="-", no_hit_color="#D55E00", label=None):
    points = sorted(points, key=lambda r: r["dt_scale_factor"])
    x = [r["dt_scale_factor"] for r in points]
    y = [float(r["hit_iteration"]) if r["status"] == "completed_hit" else np.nan for r in points]
    ax.plot(x, y, color=color, lw=1.7, marker=marker, linestyle=linestyle,
            ms=5.5, mec="white", mew=.65, zorder=4, label=label)
    for row in points:
        if row["status"] != "completed_no_hit":
            continue
        limit, factor = row["n_steps_completed"], row["dt_scale_factor"]
        ax.scatter([factor], [limit], marker="^", s=65, facecolors="none", edgecolors=no_hit_color, lw=1.5, zorder=5)
        ax.annotate("", xy=(factor, limit * 1.20), xytext=(factor, limit * 1.025),
                    arrowprops={"arrowstyle": "-|>", "color": no_hit_color, "lw": 1.2}, zorder=5)


def draw_panel(ax, points, *, compact=False, no_hit_color="#D55E00", labels=None):
    points = sorted(points, key=lambda r: r["dt_scale_factor"])
    draw_series(ax, points, no_hit_color=no_hit_color)
    ax.axvline(1, color="#777777", ls="--", lw=1, zorder=2)
    ax.set(xscale="log", yscale="log", xlabel=XLABEL, ylabel=YLABEL)
    ax.set_xlim(.008, 2.35)
    ax.set_ylim(.8, max(5, max(r["plotted_y"] for r in points) * 1.45))
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which="major", color="#d9d9d9", lw=.6, alpha=.7)
    ax.grid(which="minor", color="#e9e9e9", lw=.4, alpha=.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="major", direction="out", length=4, width=.8)
    ax.tick_params(which="minor", length=2, width=.6)
    ax.tick_params(axis="x", labelrotation=35, labelsize=8.5 if compact else 10)
    cid = points[0]["case_id"]
    dt0 = points[0]["unscaled_mean_field_delta_t"]
    labels = LABELS if labels is None else labels
    ax.set_title(labels[cid] + f"\n" + rf"$\pi/W_{{\mathrm{{MF}}}}={dt0:.6f}$ a.u.",
                 fontsize=10.5 if compact else 12.5, pad=9)


def legend_handles(include_misses, step_limit):
    handles = [Line2D([], [], color="#0072B2", marker="o", ms=5, lw=1.5, label="Chemical accuracy reached"),
               Line2D([], [], color="#777777", ls="--", lw=1, label=r"$\Delta t=\pi/W_{\mathrm{MF}}$")]
    if include_misses:
        handles.append(Line2D([], [], color="#D55E00", marker="^", mfc="none", ls="none", ms=6,
                              label=f"No hit within {step_limit} iterations"))
    return handles


def save(fig, prefix):
    for extension in ("pdf", "png"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=scan.DEFAULT_OUTDIR)
    parser.add_argument("--outdir", type=Path)
    options = parser.parse_args()
    root = scan.qrte.resolve_path(options.scan_dir).resolve()
    outdir = scan.qrte.resolve_path(options.outdir).resolve() if options.outdir else root / "loglog_plots"
    rows, args, sources, standard_rows = validated_rows(root)
    outdir.mkdir(parents=True, exist_ok=True)
    # Refresh the standard scan-results table without changing its schema.
    collector.write_csv(root / "scan_results.csv", standard_rows)
    collector.write_csv(outdir / "plot_data.csv", rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 12,
                         "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42})
    for cid in scan.CASE_IDS:
        points = [r for r in rows if r["case_id"] == cid]
        fig, ax = plt.subplots(figsize=(6.5, 5.4))
        fig.subplots_adjust(left=.15, right=.97, bottom=.34, top=.86)
        draw_panel(ax, points)
        ax.set_xlabel(XLABEL + " (exact QRTE)")
        handles = legend_handles(any(r["plotted_as_lower_bound"] for r in points), args.n_steps)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .015),
                   frameon=False, ncol=1 if len(handles) == 3 else 2, fontsize=9)
        save(fig, outdir / f"{cid}_loglog")
        collector.write_csv(outdir / f"{cid}_loglog.csv", points)
    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13.5))
    fig.subplots_adjust(left=.085, right=.98, bottom=.09, top=.94, hspace=.70, wspace=.28)
    for ax, cid in zip(axes.flat, scan.CASE_IDS):
        draw_panel(ax, [r for r in rows if r["case_id"] == cid], compact=True)
        ax.set_ylabel(YLABEL, fontsize=10.5)
        ax.set_xlabel(XLABEL, fontsize=11)
    fig.suptitle("Exact QRTE timestep scan", fontsize=16, y=.985)
    fig.legend(handles=legend_handles(True, args.n_steps), loc="lower center", bbox_to_anchor=(.5, .018),
               ncol=3, frameon=False, fontsize=10)
    save(fig, outdir / "all_systems_loglog")
    scan.qrte.write_json(outdir / "provenance.json", {
        "case_ids": list(scan.CASE_IDS), "factors": list(scan.FACTORS),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "x_axis": "logarithmic timestep factor delta_t/(pi/W_MF)", "y_axis": "logarithmic hit iteration",
        "no_hit_cases": [{k: r[k] for k in ("case_id", "dt_scale_factor", "n_steps_completed", "summary_path")}
                         for r in rows if r["plotted_as_lower_bound"]],
        "no_hit_display": "Open triangles with upward arrows at completed iteration counts; hit iteration remains blank",
        "chemical_accuracy_hartree": .0016, "evolution_mode": "sparse", "raster_dpi": 600,
        "plot_script_sha256": digest(Path(__file__)),
        "source_sha256": {relative(p): digest(p) for p in sorted(sources)},
        "new_molecular_calculations": False,
    })
    print(f"Validated {len(rows)} conditions: {Counter(r['status'] for r in rows)}")
    print(f"Saved eight individual PDF/PNG/CSV plots, combined PDF/PNG, plot_data.csv and provenance.json to {outdir}")


if __name__ == "__main__":
    main()
