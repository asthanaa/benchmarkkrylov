#!/usr/bin/env python
"""Plot N2 overlap/gap versus iterations for two QRTE timesteps and QKUD.

Includes both 1.20 A equilibrium spin cases. No SCF, QRTE, or diagonalization.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path

import plot_n2_pes_overlap_gap_iterations as base

ROOT = base.BENCHMARK_ROOT
DEFAULT_PREFIX = ROOT / "plots/n2_pes_mean_field_pi_over_w_trotter2_vs_qkud_eps0p05_overlap_gap_iterations"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def validate_row(row):
    """Match cached descriptors to Hamiltonians and recompute hits from CSVs."""
    summary_path = resolve(row["summary_path"])
    summary = base._load_json(summary_path)
    config = base._load_json(summary_path.parent / "run_config.json")
    snapshot = base._load_json(summary_path.parent / "input_snapshot.json")
    gap = base._load_json(resolve(row["gap_path"]))
    cid, spin = row["case_id"], row["spin"]
    for saved in (summary, snapshot, gap):
        if saved.get("case_id") != cid:
            raise ValueError(f"{cid}: mismatched saved case")
    if summary.get("status") != "completed" or snapshot["spin"] != spin:
        raise ValueError(f"{cid}: incomplete or mismatched spin")
    geometry = snapshot["geometry_angstrom"]
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(*geometry)))
    if (not math.isclose(distance, row["bond_length_angstrom"], abs_tol=1e-9)
            or snapshot["active_electrons"] != 6 or snapshot["active_orbitals"] != 6
            or snapshot["basis"] != "cc-pVTZ"
            or (gap["n_alpha"], gap["n_beta"]) != (3 + spin // 2, 3 - spin // 2)):
        raise ValueError(f"{cid}: geometry/active-sector mismatch")
    overlap = float(gap["overlap_weight"])
    energy_gap = float(gap["first_excited_energy_hartree"]) - float(gap["sector_ground_energy_hartree"])
    if (not 0 < overlap <= 1 or not math.isfinite(energy_gap) or energy_gap <= 0
            or not math.isclose(energy_gap, gap["gap_hartree"], rel_tol=1e-8)
            or not math.isclose(-math.log10(overlap), gap["negative_log10_overlap"], abs_tol=1e-8)):
        raise ValueError(f"{cid}: invalid saved overlap/gap")
    cache_hash = digest(gap["hamiltonian_cache_path"])
    reference = summary.get("local_reference_energy_hartree", summary["casscf_energy_hartree"])
    if abs(reference - gap["reference_energy_hartree"]) > 1e-6:
        raise ValueError(f"{cid}: method and descriptor references differ")
    method = row["method_key"]
    if summary.get("method") != method or config["evolution_mode"] != "trotter" or config["trotter_steps"] != 1:
        raise ValueError(f"{cid}: unexpected method/evolution settings")
    if row["series_key"] == "qrte_mf":
        if (summary["trotter_order"] != 2 or config["trotter_order"] != 2
                or config["overlap_tol"] != 1e-10 or config["n_steps"] != 100
                or cache_hash != summary["hamiltonian_sha256"]
                or digest(summary["hamiltonian_cache_path"]) != cache_hash):
            raise ValueError(f"{cid}: QRTE settings/cache differ from descriptors")
        width = summary["mean_field_active_bandwidth_hartree"]
        if not math.isfinite(width) or width <= 0 or not math.isclose(summary["delta_t"], math.pi / width, rel_tol=1e-12):
            raise ValueError(f"{cid}: QRTE timestep is not pi/W_MF")
        if digest(summary["source_mean_field_reference_path"]) != summary["source_mean_field_reference_sha256"]:
            raise ValueError(f"{cid}: saved HF spectrum changed")
        row.update(method="QRTE dt=pi/W_MF, Trotter order 2, one slice", delta_t=summary["delta_t"],
                   mean_field_width_hartree=width, epsilon="", trotter_order=2)
    elif row["series_key"] == "qrte_fixed":
        if (not math.isclose(config["delta_t"], .05, rel_tol=0, abs_tol=1e-14)
                or config.get("trotter_order", 1) != 1
                or not math.isclose(summary.get("delta_t", .05), .05, rel_tol=0, abs_tol=1e-14)):
            raise ValueError(f"{cid}: expected first-order QRTE dt=.05")
        saved_cache = summary.get("hamiltonian_cache_path", snapshot.get("hamiltonian_cache_path"))
        if saved_cache:
            if digest(saved_cache) != cache_hash:
                raise ValueError(f"{cid}: fixed-step QRTE cache differs from descriptors")
            cache_link = "saved Hamiltonian path and matching SHA256"
        else:
            # Legacy equilibrium runs save the shared cache directory and input,
            # but omit the individual Hamiltonian filename.
            overlap_dir = resolve(gap["overlap_run_dir"])
            overlap_input = base._load_json(overlap_dir / "input_snapshot.json")
            overlap_config = base._load_json(overlap_dir / "run_config.json")
            for key in ("symbols", "basis", "charge", "spin", "active_electrons", "active_orbitals",
                        "source_json_path", "active_orbital_labels", "active_orbital_indices"):
                if snapshot.get(key) != overlap_input.get(key):
                    raise ValueError(f"{cid}: legacy QRTE descriptor input differs at {key}")
            if (not base.np.allclose(snapshot["geometry_angstrom"], overlap_input["geometry_angstrom"],
                                     rtol=0, atol=1e-10)
                    or resolve(config["hamiltonian_cache_dir"]).resolve()
                       != resolve(gap["hamiltonian_cache_path"]).parent.resolve()
                    or config["hamiltonian_source"] != overlap_config["hamiltonian_source"]):
                raise ValueError(f"{cid}: legacy QRTE cache/input differs from descriptors")
            cache_link = "legacy shared cache directory and matching electronic/geometry input; no saved per-run cache hash"
        row.update(method="QRTE dt=0.05 a.u., Trotter order 1, one slice", delta_t=.05,
                   mean_field_width_hartree="", epsilon="", trotter_order=1,
                   cache_link_validation=cache_link)
    else:
        if config["epsilon"] != .05 or config.get("trotter_order", 1) != 1:
            raise ValueError(f"{cid}: expected original first-order QKUD epsilon=.05")
        if resolve(gap["overlap_run_dir"]).resolve() != summary_path.parent.resolve():
            raise ValueError(f"{cid}: descriptors were computed for another QKUD input")
        if "hamiltonian_cache_path" in summary and digest(summary["hamiltonian_cache_path"]) != cache_hash:
            raise ValueError(f"{cid}: QKUD cache differs from descriptors")
        row.update(method="QKUD eps=0.05, Trotter order 1, one slice", delta_t="",
                   mean_field_width_hartree="", epsilon=.05, trotter_order=1)
    history_path = summary_path.parent / "min_energy_history.csv"
    with history_path.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    errors = [float(point["min_energy_hartree"]) - reference for point in history]
    if (not history or len(history) != summary["n_steps_completed"]
            or [int(p["iteration"]) for p in history] != list(range(1, len(history) + 1))
            or not all(math.isfinite(e) for e in errors) or min(errors) < -.0016):
        raise ValueError(f"{cid}: invalid history")
    hit = next((i for i, error in enumerate(errors, 1) if abs(error) <= .0016), None)
    if hit != summary["hit_iteration"] or (hit is not None) != summary["hit_chemical_accuracy"]:
        raise ValueError(f"{cid}: history disagrees with summary hit")
    row.update(hit_chemical_accuracy=str(hit is not None).lower(),
               iteration_to_chemical_accuracy=hit if hit is not None else "",
               steps_completed=len(history), reference_energy_hartree=reference,
               final_error_hartree=errors[-1], hamiltonian_sha256=cache_hash,
               history_path=base._relative(history_path), trotter_steps=1,
               n_steps_requested=summary["n_steps_requested"],
               plot_status="chemical_accuracy_hit" if hit is not None else "lower_bound_no_hit",
               trotter_order_source="saved config" if "trotter_order" in config else "legacy first-order circuit default",
               hamiltonian_cache_path=base._relative(resolve(gap["hamiltonian_cache_path"])))


SERIES = {
    "qrte_fixed": (r"QRTE ($\Delta t=0.05$ a.u.)", {0: "^", 2: "v"}),
    "qrte_mf": (r"QRTE ($\Delta t=\pi/W_{\mathrm{MF}}$)", {0: "o", 2: "s"}),
    "qkud": (r"QKUD ($\epsilon=0.05$)", {0: "s", 2: "o"}),
}


def plot(rows, prefix):
    """Shared bond-length colors; six method/spin markers and explicit bounds."""
    plt, np = base.plt, base.np
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.linewidth": .9,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })
    fig = plt.figure(figsize=(10.8, 5.65))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, .035),
                            left=.085, right=.93, bottom=.30, top=.95, wspace=.12)
    axes = [fig.add_subplot(grid[0, 0])]
    axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
    cmap = plt.get_cmap("viridis")
    norm = base.mcolors.Normalize(min(r["bond_length_angstrom"] for r in rows),
                                 max(r["bond_length_angstrom"] for r in rows))
    panels = (("overlap_weight", r"Initial overlap, $|\langle\Phi_{\mathrm{ref}}|\Psi_0\rangle|^2$", "linear"),
              ("gap_ev", r"Spectral gap, $E_1-E_0$ (eV)", "log"))
    hits = base._hit_rows(rows)
    for index, (ax, (x_key, xlabel, scale)) in enumerate(zip(axes, panels)):
        for cid in sorted({r["case_id"] for r in hits}):
            case = [r for r in hits if r["case_id"] == cid]
            x = float(case[0][x_key])
            y = [float(r["iteration_to_chemical_accuracy"]) for r in case]
            ax.plot([x, x], [min(y), max(y)], color="#a4a4a4", lw=.65, alpha=.38, zorder=1)
        for series, (_, markers) in SERIES.items():
            for spin, marker in markers.items():
                subset = [r for r in hits if r["series_key"] == series and r["spin"] == spin]
                color = cmap(norm([r["bond_length_angstrom"] for r in subset]))
                ax.scatter([r[x_key] for r in subset],
                           [r["iteration_to_chemical_accuracy"] for r in subset],
                           marker=marker, s=66 if series == "qrte_fixed" else 57,
                           facecolors=color if series == "qkud" else "none",
                           edgecolors="#303030" if series == "qkud" else color,
                           linewidths=.5 if series == "qkud" else 1.35,
                           zorder=5 if series == "qkud" else 4)
        for row in rows:
            if row["hit_chemical_accuracy"] == "true":
                continue
            x, limit = row[x_key], row["steps_completed"]
            color = cmap(norm(row["bond_length_angstrom"]))
            # Arrow begins at the completed step count; a hit is not implied.
            ax.annotate("", xy=(x, limit * 1.25), xytext=(x, limit),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.5}, zorder=6)
            ax.plot(x, limit, marker="_", color=color, ms=9, mew=1.5, zorder=6)
        ax.set(xlabel=xlabel, xscale=scale, yscale="log")
        ax.text(.02, 1.035, f"({chr(97 + index)})", transform=ax.transAxes,
                fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(which="major", direction="out", length=4.5, width=.9)
        ax.tick_params(which="minor", length=2.6, width=.6)
        ax.grid(which="major", color="#d6d6d6", lw=.6, alpha=.65)
        ax.grid(which="minor", axis="y", color="#e5e5e5", lw=.45, alpha=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(r"$N_{\mathrm{QKrylov}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)")
    axes[0].set_ylim(1.65, max(base._plot_y(r) for r in rows) * 1.45)
    axes[0].yaxis.set_major_locator(base.LogLocator(base=10))
    axes[0].yaxis.set_major_formatter(base.LogFormatterMathtext())
    axes[0].yaxis.set_minor_locator(base.LogLocator(base=10, subs=range(2, 10)))
    axes[0].yaxis.set_minor_formatter(base.NullFormatter())
    axes[1].tick_params(labelleft=False)
    overlap = [r["overlap_weight"] for r in rows]
    axes[0].set_xlim(max(0, min(overlap) - .04), min(1, max(overlap) + .04))
    gaps = [r["gap_ev"] for r in rows]
    axes[1].set_xlim(min(gaps) * .8, max(gaps) * 1.25)
    axes[1].xaxis.set_major_formatter(base.StrMethodFormatter("{x:g}"))
    axes[1].xaxis.set_minor_formatter(base.NullFormatter())
    colorbar = fig.colorbar(base.cm.ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_subplot(grid[0, 2]))
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(r"N–N bond distance, $R$ ($\mathrm{\AA}$)", labelpad=8)
    colorbar.set_ticks((.9, 1.2, 1.5, 2., 2.4))
    colorbar.ax.yaxis.set_major_formatter(base.StrMethodFormatter("{x:.2f}"))
    colorbar.minorticks_off()
    colorbar.outline.set_linewidth(.7)
    handles = []
    for series, (label, markers) in SERIES.items():
        for spin, marker in markers.items():
            handles.append(base.Line2D([], [], marker=marker, linestyle="none", markersize=7,
                                      markerfacecolor="#5d6670" if series == "qkud" else "none",
                                      markeredgecolor="#5d6670", markeredgewidth=1.25,
                                      label=label + f", spin {spin}"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.50, .055), ncol=3,
               frameon=False, fontsize=10, columnspacing=1.9, handletextpad=.5, labelspacing=.85)
    for extension in ("pdf", "png"):
        fig.savefig(prefix.with_suffix("." + extension), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()
    prefix = resolve(args.output_prefix)
    rows = base._build_rows(
        qrte_dir=ROOT / "runs/n2_pes_qrte_mean_field_pi_over_w_trotter2",
        qkud_dir=base.DEFAULT_QKUD_DIR, gap_dir=base.DEFAULT_GAP_DIR,
        equilibrium_qrte_dir=ROOT / "runs/qrte_mean_field_active_pi_over_w_small10_trotter2",
        equilibrium_qkud_dir=base.DEFAULT_EQUILIBRIUM_QKUD_DIR,
        equilibrium_gap_dir=base.DEFAULT_EQUILIBRIUM_GAP_DIR,
        equilibrium_qrte_rerun_dir=None, include_equilibrium=True, expected_count=16, allow_missing=False)
    if len(rows) != 36 or len([row for row in rows if row["point_kind"] == "equilibrium"]) != 4:
        raise ValueError("Expected 18 geometries/spin cases and both methods, including both equilibrium spins")
    for row in rows:
        row["series_key"] = "qrte_mf" if row["method_key"] == "qrte" else "qkud"
    fixed_rows = base._build_rows(
        qrte_dir=base.DEFAULT_QRTE_DIR, qkud_dir=base.DEFAULT_QKUD_DIR, gap_dir=base.DEFAULT_GAP_DIR,
        equilibrium_qrte_dir=base.DEFAULT_EQUILIBRIUM_QRTE_DIR,
        equilibrium_qkud_dir=base.DEFAULT_EQUILIBRIUM_QKUD_DIR,
        equilibrium_gap_dir=base.DEFAULT_EQUILIBRIUM_GAP_DIR,
        equilibrium_qrte_rerun_dir=base.DEFAULT_EQUILIBRIUM_QRTE_RERUN_DIR,
        include_equilibrium=True, expected_count=16, allow_missing=False)
    for row in fixed_rows:
        if row["method_key"] == "qrte":
            rows.append({**row, "series_key": "qrte_fixed"})
    if len(rows) != 54 or len({(r["case_id"], r["series_key"]) for r in rows}) != 54:
        raise ValueError("Expected exactly 18 cases for each of the three series")
    rows.sort(key=lambda r: (r["spin"], r["bond_length_angstrom"], r["series_key"]))
    for row in rows:
        validate_row(row)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fields = base.FIELDNAMES + ["method_key", "series_key", "delta_t", "mean_field_width_hartree", "epsilon", "trotter_order",
                               "trotter_steps", "reference_energy_hartree", "final_error_hartree",
                               "hamiltonian_sha256", "hamiltonian_cache_path", "history_path",
                               "n_steps_requested", "plot_status", "trotter_order_source", "cache_link_validation"]
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, prefix)
    source_paths = {resolve(row[key]) for row in rows for key in
                    ("summary_path", "gap_path", "history_path", "hamiltonian_cache_path")}
    for row in rows:
        run_dir = resolve(row["summary_path"]).parent
        source_paths.update(run_dir / filename for filename in ("run_config.json", "input_snapshot.json"))
        summary = base._load_json(run_dir / "summary.json")
        if "source_mean_field_reference_path" in summary:
            source_paths.add(resolve(summary["source_mean_field_reference_path"]))
        if "legacy shared cache" in row.get("cache_link_validation", ""):
            gap = base._load_json(resolve(row["gap_path"]))
            source_paths.update(resolve(gap["overlap_run_dir"]) / filename
                                for filename in ("run_config.json", "input_snapshot.json"))
    audit = {"case_count": 18, "method_points": len(rows), "equilibrium_bond_angstrom": 1.2,
             "equilibrium_case_ids": list(base.EQUILIBRIUM_CASE_IDS), "new_molecular_calculations": False,
             "overlap_definition": "squared overlap of the original initial determinant with the fixed-sector ground state",
             "gap_definition": "E1-E0 in the fixed active Nalpha/Nbeta sector; no additional S^2 projection",
             "chemical_accuracy_hartree": .0016,
             "series": {key: {"label": label, "case_count": sum(r["series_key"] == key for r in rows),
                              "chemical_accuracy_hits": len(base._hit_rows([r for r in rows if r["series_key"] == key])),
                              "spin_markers": markers} for key, (label, markers) in SERIES.items()},
             "no_hit_cases": [{key: r[key] for key in ("case_id", "series_key", "steps_completed", "summary_path")}
                              for r in rows if r["hit_chemical_accuracy"] != "true"],
             "no_hit_display": "Upward arrows starting at completed iteration count; lower bounds, not chemical-accuracy hits",
             "cache_link_notes": {r["case_id"]: r["cache_link_validation"] for r in rows if "cache_link_validation" in r},
             "rendering": {"raster_dpi": 600, "pdf": "vector graphics with embedded TrueType fonts",
                           "figure_size_inches": [10.8, 5.65], "legend": "shared, below panels",
                           "legend_anchor": [.5, .055], "footnote": None,
                           "y_axis_label": r"$N_{\mathrm{QKrylov}}$ to $E_{\mathrm{chem}}$ (1.6 mHa)",
                           "overlap_axis": "linear", "gap_axis": "logarithmic", "iteration_axis": "logarithmic"},
             "plot_script_sha256": digest(Path(__file__)),
             "source_sha256": {base._relative(path): digest(path) for path in sorted(source_paths)}}
    prefix.with_suffix(".json").write_text(base.json.dumps(audit, indent=2) + "\n")
    print(f"Saved PNG/PDF/CSV/JSON: {prefix}")
    for series in SERIES:
        points = [row for row in rows if row["series_key"] == series]
        print(f"{series}: {len(base._hit_rows(points))}/{len(points)} chemical-accuracy hits")
    print("Equilibrium R=1.20 A: spin 0 and spin 2, all three series.")


if __name__ == "__main__":
    main()
