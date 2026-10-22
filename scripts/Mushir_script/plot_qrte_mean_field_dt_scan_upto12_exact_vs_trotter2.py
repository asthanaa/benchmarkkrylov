#!/usr/bin/env python
"""Overlay completed exact and second-order Trotter timestep scans.

Read and validate both saved histories before drawing either curve. Does not
launch calculations or replace the original exact-only plots.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import plot_qrte_mean_field_dt_scan_upto12_exact_loglog as style
import run_qrte_mean_field_active_dt_scan_upto12_trotter2 as trotter

scan, plt = style.scan, style.plt
DEFAULT_OUTDIR = scan.DEFAULT_OUTDIR / "exact_vs_trotter2_loglog"
PAIR_FIELDS = (
    "delta_t", "unscaled_mean_field_delta_t", "mean_field_active_bandwidth_hartree",
    "cached_reference_energy_hartree", "dataset_casscf_energy_hartree",
    "hamiltonian_sha256", "mean_field_reference_sha256", "qubits",
)


def validate_pairs(exact_rows, trotter_rows):
    def keyed(rows):
        result = {(r["case_id"], r["dt_scale_factor"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate system/timestep in comparison")
        return result
    exact, approximate = keyed(exact_rows), keyed(trotter_rows)
    if exact.keys() != approximate.keys():
        raise ValueError("Exact and Trotter grids differ")
    for key, row in exact.items():
        for field in PAIR_FIELDS:
            if row[field] != approximate[key][field]:
                raise ValueError(f"{key}: comparison differs for {field}")


def draw_overlay(ax, exact_points, trotter_points, *, compact=False, labels=None):
    style.draw_panel(ax, exact_points, compact=compact, no_hit_color="#0072B2", labels=labels)
    style.draw_series(ax, trotter_points, color="#D55E00", marker="s", linestyle="--", no_hit_color="#D55E00")
    ax.set_ylim(.8, max(5, max(r["plotted_y"] for r in exact_points + trotter_points) * 1.45))


def handles(no_hit, step_limit):
    result = [
        style.Line2D([], [], color="#0072B2", marker="o", lw=1.7, label="Exact exponential QRTE"),
        style.Line2D([], [], color="#D55E00", marker="s", ls="--", lw=1.7,
                     label="QRTE: second-order Trotter, one slice"),
        style.Line2D([], [], color="#777777", ls="--", lw=1, label=r"$\Delta t=\pi/W_{\mathrm{MF}}$"),
    ]
    if no_hit:
        result.append(style.Line2D([], [], color="#444444", marker="^", mfc="none", ls="none",
                                   label=f"No hit within {step_limit} iterations (series color)"))
    return result


def main(*, exact_module=None, trotter_module=None, labels=None):
    exact_driver = scan if exact_module is None else exact_module
    trotter_driver = trotter if trotter_module is None else trotter_module
    case_ids = exact_driver.CASE_IDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-scan-dir", type=Path, default=exact_driver.DEFAULT_OUTDIR)
    parser.add_argument("--trotter-scan-dir", type=Path, default=trotter_driver.DEFAULT_OUTDIR)
    parser.add_argument("--outdir", type=Path, default=exact_driver.DEFAULT_OUTDIR / "exact_vs_trotter2_loglog")
    options = parser.parse_args()
    exact_root = scan.qrte.resolve_path(options.exact_scan_dir).resolve()
    trotter_root = scan.qrte.resolve_path(options.trotter_scan_dir).resolve()
    exact, exact_args, exact_sources, exact_standard = style.validated_rows(exact_root, scan_module=exact_driver)
    approximate, trotter_args, trotter_sources, trotter_standard = style.validated_rows(trotter_root, scan_module=trotter_driver)
    validate_pairs(exact, approximate)
    for name in ("n_steps", "overlap_tol", "max_workers"):
        if getattr(exact_args, name) != getattr(trotter_args, name):
            raise ValueError(f"Comparison stopping/worker settings differ for {name}")
    outdir = scan.qrte.resolve_path(options.outdir).resolve()
    if outdir == exact_root / "loglog_plots":
        raise ValueError("Use a separate comparison output directory to preserve exact-only plots")
    outdir.mkdir(parents=True, exist_ok=True)
    style.collector.write_csv(exact_root / "scan_results.csv", exact_standard)
    style.collector.write_csv(trotter_root / "scan_results.csv", trotter_standard)
    rows = [{**r, "series": series} for series, data in (("exact", exact), ("trotter2", approximate)) for r in data]
    style.collector.write_csv(outdir / "plot_data.csv", rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 12,
                         "axes.linewidth": .8, "pdf.fonttype": 42, "ps.fonttype": 42})
    for cid in case_ids:
        a, b = [r for r in exact if r["case_id"] == cid], [r for r in approximate if r["case_id"] == cid]
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        fig.subplots_adjust(left=.15, right=.97, bottom=.37, top=.86)
        draw_overlay(ax, a, b, labels=labels)
        fig.legend(handles=handles(any(r["plotted_as_lower_bound"] for r in a + b), exact_args.n_steps),
                   loc="lower center", bbox_to_anchor=(.54, .015), frameon=False, ncol=1, fontsize=9)
        style.save(fig, outdir / f"{cid}_loglog")
        style.collector.write_csv(outdir / f"{cid}_loglog.csv", [r for r in rows if r["case_id"] == cid])
    panel_rows = math.ceil(len(case_ids) / 2)
    fig, axes = plt.subplots(panel_rows, 2, figsize=(11.5, max(4.8, 3.45 * panel_rows)), squeeze=False)
    fig.subplots_adjust(left=.085, right=.98, bottom=.25 if panel_rows == 1 else .11,
                        top=.78 if panel_rows == 1 else .94, hspace=.70, wspace=.28)
    for ax, cid in zip(axes.flat, case_ids):
        draw_overlay(ax, [r for r in exact if r["case_id"] == cid],
                     [r for r in approximate if r["case_id"] == cid], compact=True, labels=labels)
        ax.set_ylabel(style.YLABEL, fontsize=10.5)
        ax.set_xlabel(style.XLABEL, fontsize=11)
    for ax in list(axes.flat)[len(case_ids):]:
        ax.set_visible(False)
    fig.suptitle("QRTE timestep scan: exact vs second-order Trotter", fontsize=15, y=.985)
    fig.legend(handles=handles(any(r["plotted_as_lower_bound"] for r in rows), exact_args.n_steps),
               loc="lower center", bbox_to_anchor=(.5, .012), ncol=2, frameon=False, fontsize=9.5)
    style.save(fig, outdir / "all_systems_loglog")
    scan.qrte.write_json(outdir / "provenance.json", {
        "case_ids": list(case_ids), "factors": list(exact_driver.FACTORS), "comparison_fields": list(PAIR_FIELDS),
        "status_counts": {series: dict(Counter(r["status"] for r in data))
                          for series, data in (("exact", exact), ("trotter2", approximate))},
        "evolution_modes": {"exact": "sparse", "trotter2": "trotter, order 2, one slice"},
        "chemical_accuracy_hartree": .0016, "axes": "log-log", "raster_dpi": 600,
        "no_hit_display": "Series-colored open triangles/upward arrows; blank hit iterations in CSV",
        "source_sha256": {style.relative(p): style.digest(p) for p in sorted(exact_sources | trotter_sources)},
        "plot_script_sha256": style.digest(Path(__file__)), "new_molecular_calculations": False,
    })
    print(f"Saved {len(case_ids)} individual overlays and the combined PDF/600-dpi PNG comparison to {outdir}")


if __name__ == "__main__":
    main()
