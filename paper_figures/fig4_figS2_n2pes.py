"""Fig. 4 (iterations vs internuclear distance along the two N2 curves) and Fig. S2
(iterations vs squared initial overlap / fixed-sector gap along the same curves),
both from plots/n2_pes_mean_field_pi_over_w_trotter2_vs_qkud_eps0p05_overlap_gap_iterations.csv.

usage: python paper_figures/fig4_figS2_n2pes.py [repo_root] [out_dir]   (defaults: repo root, paper_figures/output)
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, StrMethodFormatter

import figstyle as st

HERE = Path(__file__).resolve().parent
repo = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "output"
out.mkdir(parents=True, exist_ok=True)
csv_path = repo / "plots/n2_pes_mean_field_pi_over_w_trotter2_vs_qkud_eps0p05_overlap_gap_iterations.csv"
rows = list(csv.DictReader(csv_path.open()))
assert len(rows) == 54, len(rows)
for r in rows:
    r["R"] = float(r["bond_length_angstrom"]); r["spin"] = int(r["spin"])
    r["hit"] = str(r["hit_chemical_accuracy"]).lower() == "true"
    r["y"] = int(r["iteration_to_chemical_accuracy"]) if r["hit"] else int(r["steps_completed"])
    r["overlap"] = float(r["overlap_weight"]); r["gap"] = float(r["gap_ev"])
    assert r["hit"] or r["steps_completed"] == "100"
SERIES = ("qrte_fixed", "qrte_mf", "qkud")
SPINS = {0: "spin 0", 2: "spin 2"}
st.apply()


def y_axis(ax):
    ax.set_yscale("log")
    ax.set_ylim(1.6, 260)
    ax.yaxis.set_major_locator(FixedLocator((2, 3, 5, 10, 20, 50, 100)))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())


# Fig. 4 ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(st.TEXTWIDTH, 2.9), sharey=True)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.165, top=0.90, wspace=0.08)
for ax, spin, letter in zip(axes, (0, 2), "ab"):
    for key in SERIES:
        sel = sorted([r for r in rows if r["series_key"] == key and r["spin"] == spin], key=lambda r: r["R"])
        x = [r["R"] for r in sel]; y = [r["y"] for r in sel]
        ax.plot(x, y, color=st.COLORS[key], ls="-", lw=1.1, marker="none", zorder=3)
        ax.plot([r["R"] for r in sel if r["hit"]], [r["y"] for r in sel if r["hit"]], ls="none",
                marker=st.MARKERS[key], ms=st.MS, color=st.COLORS[key], mec="white", mew=0.5, zorder=5, label=st.LABELS[key])
        for r in sel:
            if not r["hit"]:
                ax.plot(r["R"], r["y"], ls="none", marker=st.MARKERS[key], ms=st.MS, mfc="white", mec=st.COLORS[key], mew=0.9, zorder=6)
                st.no_hit_arrow(ax, r["R"], r["y"], st.COLORS[key])
    y_axis(ax)
    ax.set_xlim(0.8, 2.5)
    ax.set_xlabel(r"Internuclear distance, $R$ (Å)")
    st.grid(ax)
    st.panel_label(ax, letter, r"N$_2$/cc-pVTZ, " + ("$2S=0$ (singlet)" if spin == 0 else "$2S=2$ (triplet)"))
axes[0].set_ylabel(st.YLABEL)
handles = [st.series_handle(k) for k in SERIES] + [st.no_hit_handle("Could not find solution\nin 100 iterations")]
axes[0].set_ylim(1.6, 500)
axes[0].legend(handles=handles, loc="upper left", handlelength=1.2, borderaxespad=0.4)
st.save(fig, out / "fig4_n2pes_distance")
plt.close(fig)

# Fig. S2 --------------------------------------------------------------------
fig = plt.figure(figsize=(st.TEXTWIDTH, 3.5))
grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.028), left=0.075, right=0.925, bottom=0.285, top=0.92, wspace=0.08)
axes = [fig.add_subplot(grid[0, 0])]
axes.append(fig.add_subplot(grid[0, 1], sharey=axes[0]))
cax = fig.add_subplot(grid[0, 2])
cmap = plt.get_cmap("viridis")
norm = mcolors.Normalize(min(r["R"] for r in rows), max(r["R"] for r in rows))
panels = (("overlap", r"Initial overlap, $|\langle\Phi_{\mathrm{ref}}|\Psi_0\rangle|^2$", "linear"),
          ("gap", r"Spectral gap, $E_1-E_0$ (eV)", "log"))
for ax, (xkey, xlabel, scale), letter in zip(axes, panels, "ab"):
    for key in SERIES:
        for spin in (0, 2):
            sel = [r for r in rows if r["series_key"] == key and r["spin"] == spin]
            colors = [cmap(norm(r["R"])) for r in sel]
            ax.scatter([r[xkey] for r in sel], [r["y"] for r in sel], marker=st.MARKERS[key], s=st.SCATTER_S + 4,
                       facecolors=colors if spin == 0 else "none", edgecolors="white" if spin == 0 else colors,
                       linewidths=0.5 if spin == 0 else 1.0, zorder=5)
            for r in sel:
                if not r["hit"]:
                    st.no_hit_arrow(ax, r[xkey], r["y"], cmap(norm(r["R"])))
    y_axis(ax)
    ax.set(xscale=scale, xlabel=xlabel)
    st.grid(ax)
    st.panel_label(ax, letter)
axes[0].set_ylabel(st.YLABEL)
axes[0].set_xlim(0.08, 1.0)
axes[1].set_xlim(0.05, 25)
axes[1].xaxis.set_major_locator(LogLocator(base=10))
axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
axes[1].xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
axes[1].xaxis.set_minor_formatter(NullFormatter())
plt.setp(axes[1].get_yticklabels(), visible=False)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"N–N distance, $R$ (Å)")
cb.set_ticks((0.9, 1.2, 1.5, 2.0, 2.4))
cb.ax.tick_params(length=2.5)
mid = cmap(0.55)
row1 = [Line2D([], [], ls="none", marker=st.MARKERS[key], ms=st.MS + 0.5, mfc=mid, mec="white", mew=0.5, label=st.LABELS[key])
        for key in SERIES]
row2 = [Line2D([], [], ls="none", marker="o", ms=st.MS + 0.5, mfc=mid, mec="white", mew=0.5, label="Filled: spin 0 ($2S=0$)"),
        Line2D([], [], ls="none", marker="o", ms=st.MS + 0.5, mfc="white", mec=mid, mew=1.0, label="Open: spin 2 ($2S=2$)"),
        st.no_hit_handle()]
fig.legend(handles=row1, loc="lower center", bbox_to_anchor=(0.5, 0.075), ncol=3, columnspacing=2.0, handlelength=1.2)
fig.legend(handles=row2, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, columnspacing=2.0, handlelength=1.2)
st.save(fig, out / "figS2_n2pes_overlap_gap")
print("ok; no-hit rows:", [(r["series_key"], r["spin"], r["R"]) for r in rows if not r["hit"]])
