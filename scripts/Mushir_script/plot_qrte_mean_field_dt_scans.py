#!/usr/bin/env python
"""Combine the two collected QRTE timestep scans; read saved CSV data only.

Refresh the individual collectors first to validate saved histories/settings.
Writes an annotated heatmap, merged results, and a table of best tested dts.
"""

import csv
import hashlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/benchmarkkrylov-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

import collect_qrte_mean_field_active_dt_scan4 as collect
from collect_qrte_mean_field_active_dt_scan_beh2_hchains import LABELS as CHAIN_LABELS

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "plots/qrte_mean_field_dt_scans"
SOURCES = (ROOT / "runs/qrte_mean_field_active_dt_scan4_trotter2/scan_results.csv",
           ROOT / "runs/qrte_mean_field_active_dt_scan_beh2_hchains_trotter2/scan_results.csv")


def main():
    rows = []
    for source in SOURCES:
        with source.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 77 or any(row["status"] != "completed_hit" for row in rows):
        raise ValueError("This completed-scan comparison requires all 77 validated hits; refresh collectors first")
    cases = list(dict.fromkeys(row["case_id"] for row in rows))
    factors = list(collect.scan.FACTORS)
    labels = {**collect.LABELS, **CHAIN_LABELS}
    table, best_rows = [], []
    for cid in cases:
        points = [row for row in rows if row["case_id"] == cid]
        if len(points) != 7 or [float(row["dt_scale_factor"]) for row in points] != factors:
            raise ValueError(f"Incomplete factor grid for {cid}")
        values = [int(row["hit_iteration"]) for row in points]
        table.append(values)
        best = min(values)
        dt0 = float(points[0]["unscaled_mean_field_delta_t"])
        best_rows.append({"case_id": cid, "label": labels[cid], "dt0_atomic_units": dt0,
                          "iterations_at_dt0": values[factors.index(1)], "best_iterations": best,
                          "best_tested_factors": ";".join(f"{factor:g}" for factor, value in zip(factors, values) if value == best),
                          "best_tested_dt_atomic_units": ";".join(f"{factor * dt0:.12g}" for factor, value in zip(factors, values) if value == best),
                          "iterations_at_2dt0": values[-1]})
    OUTDIR.mkdir(parents=True, exist_ok=True)
    collect.write_csv(OUTDIR / "scan_results.csv", rows)
    collect.write_csv(OUTDIR / "best_timesteps.csv", best_rows)
    values = np.asarray(table)
    fig, ax = plt.subplots(figsize=(11.2, 7.5), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu", norm=LogNorm(vmin=1, vmax=values.max()))
    for i, line in enumerate(values):
        for j, value in enumerate(line):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=12,
                    color="white" if im.norm(value) > .62 else "#17212b",
                    fontweight="bold" if value == line.min() else "normal")
            if value == line.min():
                ax.add_patch(Rectangle((j - .46, i - .43), .92, .86, fill=False,
                                       edgecolor="#d55e00", linewidth=1.7))
    centre = factors.index(1)
    ax.axvline(centre - .5, color="gray", linestyle="--", linewidth=1)
    ax.axvline(centre + .5, color="gray", linestyle="--", linewidth=1)
    ax.axhline(3.5, color="white", linewidth=3)
    ax.set_xticks(range(len(factors)), [f"{factor:g}" for factor in factors])
    ax.set_yticks(range(len(cases)), [labels[cid] for cid in cases])
    ax.set_xlabel("Timestep factor Δt / dt₀, with dt₀ = π/W_MF", fontsize=12)
    ax.set_title("QRTE iterations to chemical accuracy (1.6 mHa)\n"
                 "Second-order Trotter · one slice · all 77 conditions converged", fontsize=14, pad=15)
    bar = fig.colorbar(im, ax=ax, pad=.025, shrink=.85)
    bar.set_ticks([1, 2, 4, 8, 16, 32, 43], labels=[1, 2, 4, 8, 16, 32, 43])
    bar.set_label("Iterations (logarithmic color scale)")
    fig.supxlabel("Orange boxes: minimum iteration count among the seven tested timesteps for each system", fontsize=10)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTDIR / f"iterations_heatmap.{suffix}", dpi=300)
    plt.close(fig)
    collect.scan.qrte.write_json(OUTDIR / "provenance.json", {
        "source_csv_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in SOURCES},
        "cases": cases, "factors": factors, "completed_hits": len(rows),
        "chemical_accuracy_hartree": .0016, "trotter_order": 2, "trotter_steps": 1,
        "new_molecular_calculations": False,
        "interpretation": "Best refers to iteration count at tested points only; no wall-time or exact-width optimization is inferred."})
    print(f"Saved combined heatmap, CSV tables, and provenance to {OUTDIR}")
    print("Factors:", factors)
    print("Sum of iteration counts:", values.sum(axis=0).tolist())


if __name__ == "__main__":
    main()
