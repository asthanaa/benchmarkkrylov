#!/usr/bin/env python
"""Prepare scan manifests or collect saved QRTE timestep-scan results and plot.

Never launches molecular calculations. Pending, failed, and completed no-hit
conditions remain explicit; missing hit iterations are never replaced by 100.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import run_qrte_mean_field_active_dt_scan4 as scan

LABELS = {scan.CASE_IDS[0]: "N₂/cc-pVTZ, spin 0", scan.CASE_IDS[1]: "N₂/cc-pVTZ, spin 2",
          scan.CASE_IDS[2]: "Al₃⁻", scan.CASE_IDS[3]: "H10/STO-3G"}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plan_rows(tasks):
    return [{"task_index": task["task_index"], "case_id": task["case"]["case_id"],
             "dt_scale_factor": task["factor"], "mean_field_active_bandwidth_hartree": task["width"],
             "unscaled_mean_field_delta_t": task["base_delta_t"], "delta_t": task["delta_t"],
             "trotter_order": task.get("trotter_order", 2), "trotter_steps": task.get("trotter_steps", 1),
             **({"evolution_mode": "sparse", "trotter_parameters_apply": False}
                if task.get("evolution_mode") == "sparse" else {}),
             "cached_reference_energy_hartree": task["case"]["local_reference_energy_hartree"],
             "dataset_casscf_energy_hartree": task["case"]["dataset_casscf_energy_hartree"],
             "hamiltonian_sha256": task["case"]["hamiltonian_sha256"],
             "mean_field_reference_sha256": task["case"]["mean_field_reference_sha256"],
             "output_directory": str(task["outdir"] / task["case"]["case_id"])} for task in tasks]


def write_plan(tasks, args):
    rows = plan_rows(tasks)
    plan = {"case_ids": list(dict.fromkeys(task["case"]["case_id"] for task in tasks)),
            "factors": list(dict.fromkeys(task["factor"] for task in tasks)), "task_count": len(tasks),
            "n_steps": args.n_steps, "overlap_tol": args.overlap_tol, "max_workers": args.max_workers,
            "trotter_order": args.trotter_order, "trotter_steps": args.trotter_steps, "chemical_accuracy_hartree": 0.0016,
            "new_molecular_calculations": False, "tasks": rows}
    if args.evolution_mode == "sparse":
        plan.update(evolution_mode="sparse", propagation="expm_multiply(-1j * delta_t * H, state)",
                    trotter_parameters_apply=False,
                    trotter_parameter_note="Order and slice values are unused API defaults; no product formula is constructed")
    path = args.outdir / "scan_plan.json"
    if path.exists() and scan.qrte.load_json(path) != plan:
        raise ValueError("Existing scan plan differs; use a separate --outdir for different settings or inputs")
    scan.qrte.write_json(path, plan)
    write_csv(args.outdir / "scan_plan.csv", rows)


def validate_summary(summary, task, args):
    case = task["case"]
    for field in ("case_id", "hamiltonian_sha256", "mean_field_reference_sha256"):
        if summary.get(field) != case[field]:
            raise ValueError(f"Saved result differs for {field}")
    if (summary.get("evolution_mode"), summary.get("trotter_order"), summary.get("trotter_steps")) != (
            args.evolution_mode, args.trotter_order, args.trotter_steps):
        raise ValueError("Saved result has incompatible propagation settings")
    if summary.get("n_steps_requested") != args.n_steps or summary.get("overlap_tol") != args.overlap_tol:
        raise ValueError("Saved result has incompatible stopping settings")
    if not math.isclose(summary["delta_t"], task["delta_t"], rel_tol=1e-12):
        raise ValueError("Saved result has an incompatible timestep")
    if abs(summary["local_reference_energy_hartree"] - case["local_reference_energy_hartree"]) > 1e-10:
        raise ValueError("Saved result uses a different energy reference")


def result_rows(tasks, args):
    rows = []
    for task, planned in zip(tasks, plan_rows(tasks)):
        path = Path(planned["output_directory"]) / "summary.json"
        row = {**planned, "status": "pending", "hit_chemical_accuracy": "", "hit_iteration": "",
               "n_steps_completed": "", "final_error_hartree": "", "summary_path": str(path), "error": ""}
        if path.exists():
            try:
                summary = scan.qrte.load_json(path)
                row["status"] = summary.get("status", "unknown")
                if row["status"] == "completed":
                    validate_summary(summary, task, args)
                    with (path.parent / "min_energy_history.csv").open(newline="") as handle:
                        history = list(csv.DictReader(handle))
                    if not history or len(history) != summary["n_steps_completed"]:
                        raise ValueError("Saved history length differs from summary")
                    errors = [float(point["min_energy_hartree"]) - task["case"]["local_reference_energy_hartree"]
                              for point in history]
                    if not all(math.isfinite(error) for error in errors) or min(errors) < -0.0016:
                        raise ValueError("Saved history has nonfinite or unphysical reference errors")
                    hit = next((int(point["iteration"]) for point, error in zip(history, errors)
                                if abs(error) <= 0.0016), None)
                    if hit != summary["hit_iteration"] or (hit is not None) != summary["hit_chemical_accuracy"]:
                        raise ValueError("Saved hit flag disagrees with the history")
                    row.update(status="completed_hit" if hit is not None else "completed_no_hit",
                               hit_chemical_accuracy=str(hit is not None).lower(),
                               hit_iteration=hit if hit is not None else "", n_steps_completed=len(history),
                               final_error_hartree=errors[-1])
                else:
                    row["error"] = summary.get("error", "")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                row.update(status="invalid_result", error=str(exc))
        rows.append(row)
    return rows


def plot_results(rows, args, *, case_ids=None, labels=None, loglog=False):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchmarkkrylov-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator
    import numpy as np

    case_ids = list(dict.fromkeys(row["case_id"] for row in rows)) if case_ids is None else case_ids
    labels = LABELS if labels is None else labels
    panel_rows = math.ceil(len(case_ids) / 2)
    fig, axes = plt.subplots(panel_rows, 2, figsize=(10, 3.5 * panel_rows),
                             constrained_layout=True, squeeze=False)
    for ax, cid in zip(axes.flat, case_ids):
        points = sorted((row for row in rows if row["case_id"] == cid), key=lambda r: r["dt_scale_factor"])
        factors = [row["dt_scale_factor"] for row in points]
        x = [row["dt_scale_factor"] for row in points]
        y = [row["hit_iteration"] if row["status"] == "completed_hit" else np.nan for row in points]
        ax.plot(x, y, "o-", color="#0072B2", label="Chemical accuracy reached")
        misses = [row for row in points if row["status"] == "completed_no_hit"]
        if misses:
            ax.scatter([row["dt_scale_factor"] for row in misses], [args.n_steps] * len(misses),
                       marker="^", facecolors="none", edgecolors="#D55E00", s=70,
                       label=f"No hit within {args.n_steps} iterations")
        ax.axvline(1, color="gray", linestyle="--", linewidth=0.9, label="π/W estimate")
        ax.set_title(f"{labels.get(cid, cid)}\ndt₀ = {points[0]['unscaled_mean_field_delta_t']:.6f} a.u.")
        ax.set_xlabel("Timestep factor Δt / dt₀")
        ax.set_ylabel("Iterations to chemical accuracy")
        if loglog:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(factors, labels=[f"{factor:g}" for factor in factors])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}" if value >= 1 else ""))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(bottom=0.8)
        else:
            ax.set_xticks(factors)
        if len(factors) > 7:
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        if not loglog:
            ax.set_ylim(bottom=0)
        if not any(math.isfinite(value) for value in y) and not misses:
            ax.set_ylim(0.8 if loglog else 0, 10)
            ax.text(0.5, 0.5, "Awaiting scan results", transform=ax.transAxes, ha="center", color="gray")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(case_ids):]:
        ax.set_visible(False)
    hits = sum(row["status"] == "completed_hit" for row in rows)
    misses = sum(row["status"] == "completed_no_hit" for row in rows)
    scale_label = " · log–log axes" if loglog else ""
    propagation = "exact exponential" if args.evolution_mode == "sparse" else "second-order Trotter, one slice"
    fig.suptitle(f"QRTE timestep sensitivity · {propagation}{scale_label}\n"
                 f"{hits} hits; {misses} completed no-hit conditions; "
                 f"{len(rows) - hits - misses} pending/failed/invalid", fontsize=12)
    filename = "iterations_vs_dt_scale_loglog" if loglog else "iterations_vs_dt_scale"
    for suffix in ("png", "pdf"):
        fig.savefig(args.outdir / f"{filename}.{suffix}", dpi=300)
    plt.close(fig)


def main(*, scan_module=scan, labels=None, prepare_tasks=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=scan_module.DEFAULT_OUTDIR)
    parser.add_argument("--manifest", type=Path, default=scan_module.qrte.DEFAULT_MANIFEST)
    parser.add_argument("--cache-manifest", type=Path, default=scan_module.qrte.DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--overlap-tol", type=float, default=1e-10)
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--loglog", action="store_true", help="Save a separate plot with both axes logarithmic")
    options = parser.parse_args()
    args = scan_module.parse_args(["--outdir", str(options.outdir), "--manifest", str(options.manifest),
                            "--cache-manifest", str(options.cache_manifest), "--n-steps", str(options.n_steps),
                            "--overlap-tol", str(options.overlap_tol), "--max-workers", str(options.max_workers)])
    tasks = scan_module.load_tasks(args)
    write_plan(tasks, args)
    if prepare_tasks is not None:
        prepare_tasks(tasks, args)
    print(f"Validated and recorded {len(tasks)} scan conditions in {args.outdir / 'scan_plan.csv'}")
    if options.prepare_only:
        return
    rows = result_rows(tasks, args)
    write_csv(args.outdir / "scan_results.csv", rows)
    plot_results(rows, args, case_ids=scan_module.CASE_IDS, labels=labels, loglog=options.loglog)
    for status in sorted({row["status"] for row in rows}):
        print(f"{status}: {sum(row['status'] == status for row in rows)}")
    filename = "iterations_vs_dt_scale_loglog" if options.loglog else "iterations_vs_dt_scale"
    print(f"Saved scan_results.csv and {filename}.png/.pdf to {args.outdir}")


if __name__ == "__main__":
    main()
