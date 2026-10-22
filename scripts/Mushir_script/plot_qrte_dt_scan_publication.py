#!/usr/bin/env python
"""Publication figure: eight exact/Trotter timestep scans, including hexatriene.

Read the validated histories; omit both BeH2 cases. Export vector artwork,
600-dpi PNG, selected plot data and source fingerprints.
"""

import argparse
from collections import Counter
from pathlib import Path

import plot_qrte_mean_field_dt_scan_upto12_exact_loglog as style
import plot_qrte_mean_field_dt_scan_upto12_exact_vs_trotter2 as comparison
import run_qrte_mean_field_active_dt_scan_upto12_exact as original_exact
import run_qrte_mean_field_active_dt_scan_upto12_trotter2 as original_trotter
import run_qrte_mean_field_dt_scan_hexatriene_exact as hex_exact
import run_qrte_mean_field_dt_scan_hexatriene_trotter2 as hex_trotter

plt, np = style.plt, style.np
ROOT = original_exact.qrte.ROOT
DEFAULT_PREFIX = ROOT / "plots/qrte_dt_scan_final/qrte_exact_vs_trotter2_loglog"
XLABEL = r"$\delta t/\delta t_{\mathrm{MF}}$"
CASES = (
    "N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2",
    "hexatriene_equi_cc_pVTZ_0", "hexatriene_equi_cc_pVTZ_2",
    "H4_near_equi_sto3g_0", "H6_near_equi_sto3g_0",
    "H6_near_equi_cc_pVTZ_0", "H6_near_equi_cc_pVTZ_2",
)
LABELS = (
    "N₂/cc-pVTZ, spin 0", "N₂/cc-pVTZ, spin 2",
    "Hexatriene/cc-pVTZ, spin 0", "Hexatriene/cc-pVTZ, spin 2",
    "H₄/STO-3G, spin 0", "H₆/STO-3G, spin 0",
    "H₆/cc-pVTZ, spin 0", "H₆/cc-pVTZ, spin 2",
)
SERIES = {
    "exact": {"color": "#0072B2", "marker": "o", "linestyle": "-", "markersize": 5.5},
    "trotter2": {"color": "#D55E00", "marker": "s", "linestyle": "--", "markersize": 3.1},
}


def load_selected():
    rows, sources, settings = [], set(), []
    for driver, series in ((original_exact, "exact"), (original_trotter, "trotter2"),
                           (hex_exact, "exact"), (hex_trotter, "trotter2")):
        data, args, _, _ = style.validated_rows(driver.DEFAULT_OUTDIR, scan_module=driver)
        settings.append((args.n_steps, args.overlap_tol, args.max_workers))
        sources.add(driver.DEFAULT_OUTDIR / "scan_plan.json")
        for row in data:
            if row["case_id"] not in CASES:
                continue
            rows.append({**row, "series": series, "panel": chr(97 + CASES.index(row["case_id"])),
                         "scan_root": style.relative(driver.DEFAULT_OUTDIR)})
            for field in ("summary_path", "history_path", "config_path", "hamiltonian_cache_path", "mean_field_reference_path"):
                sources.add(ROOT / row[field])
            run_dir = (ROOT / row["summary_path"]).parent
            sources.add(run_dir / "input_snapshot.json")
            if (run_dir / "reused_from.json").exists():
                sources.add(run_dir / "reused_from.json")
    if len(set(settings)) != 1:
        raise ValueError("The four source scans have different stopping/worker settings")
    if len(rows) != 176 or any("BeH2" in r["case_id"] for r in rows):
        raise ValueError("Expected 176 selected conditions and no BeH2 points")
    if Counter((r["case_id"], r["series"]) for r in rows) != Counter({(cid, series): 11 for cid in CASES for series in SERIES}):
        raise ValueError("Incomplete or duplicate selected factor grid")
    comparison.validate_pairs([r for r in rows if r["series"] == "exact"],
                              [r for r in rows if r["series"] == "trotter2"])
    rows.sort(key=lambda r: (CASES.index(r["case_id"]), list(SERIES).index(r["series"]), r["dt_scale_factor"]))
    return rows, sources, settings[0]


def draw_panel(ax, points, index):
    for series, spec in SERIES.items():
        selected = [r for r in points if r["series"] == series]
        x = [r["dt_scale_factor"] for r in selected]
        y = [float(r["hit_iteration"]) if r["status"] == "completed_hit" else np.nan for r in selected]
        ax.plot(x, y, color=spec["color"], marker=spec["marker"], linestyle=spec["linestyle"],
                lw=1.3, ms=spec["markersize"], mfc="white" if series == "exact" else spec["color"],
                mec=spec["color"], mew=.95 if series == "exact" else .5, zorder=4 if series == "exact" else 5)
        misses = [r for r in selected if r["status"] == "completed_no_hit"]
        for row in misses:
            ax.plot(row["dt_scale_factor"], row["n_steps_completed"], marker="^", ls="none",
                    ms=7 if series == "exact" else 4.5, mfc="none", mec=spec["color"], mew=1.05, zorder=6)
    # One upward arrow per coincident bound, with both series-colored markers.
    bounds = {(r["dt_scale_factor"], r["n_steps_completed"]) for r in points if r["plotted_as_lower_bound"]}
    for factor, limit in bounds:
        ax.annotate("", xy=(factor, limit * 1.32), xytext=(factor, limit * 1.13),
                    arrowprops={"arrowstyle": "-|>", "color": "#444444", "lw": .9, "mutation_scale": 7}, zorder=6)
    ax.axvline(1, color="#777777", ls=(0, (3, 3)), lw=.85, zorder=2)
    ax.set(xscale="log", yscale="log", xlim=(.008, 2.35), ylim=(.8, 150))
    ax.xaxis.set_major_locator(style.FixedLocator(style.XTICKS))
    ax.xaxis.set_major_formatter(style.StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_locator(style.LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(style.NullFormatter())
    ax.yaxis.set_major_locator(style.FixedLocator((1, 3, 10, 30, 100)))
    ax.yaxis.set_major_formatter(style.StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(style.LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(style.NullFormatter())
    ax.grid(False, which="both")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(which="major", direction="out", length=3, width=.7, labelsize=8)
    ax.tick_params(which="minor", direction="out", length=1.8, width=.45)
    ax.set_title(f"({chr(97 + index)})  {LABELS[index]}", loc="left", fontsize=9, pad=18)
    dt0 = points[0]["unscaled_mean_field_delta_t"]
    ax.text(0, 1.015, rf"$\delta t_{{\mathrm{{MF}}}}={dt0:.4f}$ a.u.", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=7.2, color="#505050")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()
    prefix = original_exact.qrte.resolve_path(args.output_prefix).resolve()
    rows, sources, settings = load_selected()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    style.collector.write_csv(prefix.with_suffix(".csv"), rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.linewidth": .7,
                         "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "savefig.dpi": 600})
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 8.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.985, bottom=.175, top=.955, hspace=.36, wspace=.12)
    for index, (ax, cid) in enumerate(zip(axes.flat, CASES)):
        draw_panel(ax, [r for r in rows if r["case_id"] == cid], index)
        if index >= 6:
            ax.tick_params(axis="x", labelrotation=35, labelsize=7.5)
    fig.supxlabel(XLABEL, fontsize=11, y=.10)
    fig.supylabel(style.YLABEL, fontsize=11, x=.012)
    handles = [
        style.Line2D([], [], color="#0072B2", marker="o", mfc="white", ms=5.5, lw=1.3,
                     label="Exact exponential QRTE"),
        style.Line2D([], [], color="#D55E00", marker="s", ms=3.5, ls="--", lw=1.3,
                     label="Second-order Trotter QRTE (one slice)"),
        style.Line2D([], [], color="#777777", ls=(0, (3, 3)), lw=.85,
                     label=r"$\delta t=\delta t_{\mathrm{MF}}$"),
        style.Line2D([], [], color="#444444", marker="^", mfc="none", ls="none", ms=5.5,
                     label=f"No hit within {settings[0]} iterations"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .017), ncol=2,
               frameon=False, fontsize=8, handlelength=2.4, handletextpad=.7,
               columnspacing=2.2, labelspacing=.8)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    original_exact.qrte.write_json(prefix.with_suffix(".json"), {
        "case_ids": list(CASES), "panel_labels": list(LABELS), "factors": list(original_exact.FACTORS),
        "excluded_case_ids": ["BeH2_equi_cc_pVTZ_0", "BeH2_equi_cc_pVTZ_2"],
        "row_count": len(rows), "status_counts": {series: dict(Counter(r["status"] for r in rows if r["series"] == series))
                                                   for series in SERIES},
        "chemical_accuracy_hartree": .0016, "max_iterations": settings[0], "overlap_tol": settings[1],
        "no_hit_rows": [{key: r[key] for key in ("case_id", "series", "dt_scale_factor", "n_steps_completed", "summary_path")}
                        for r in rows if r["plotted_as_lower_bound"]],
        "no_hit_display": "Nested open triangles at completed-step bounds and upward arrows; blank hit iterations in CSV",
        "layout": "4 by 2, shared logarithmic axes and shared legend; no global title or footnote",
        "figure_size_inches": [7.4, 8.7], "y_limits": [.8, 150], "raster_dpi": 600,
        "x_axis_label": XLABEL, "x_axis_units": "dimensionless",
        "mean_field_timestep_definition": "delta t_MF = pi/W_MF",
        "timestep_annotation_location": "above axes, left aligned, below panel title", "interior_grid": False,
        "formats": ["PDF with embedded TrueType fonts", "SVG with editable text", "PNG"],
        "source_sha256": {style.relative(p): style.digest(p) for p in sorted(sources)},
        "plot_script_sha256": style.digest(Path(__file__)), "new_molecular_calculations": False,
    })
    print(f"Validated and plotted {len(rows)} conditions across eight systems, excluding BeH2")
    for extension in ("pdf", "png", "svg", "csv", "json"):
        print(prefix.with_suffix("." + extension))


if __name__ == "__main__":
    main()
