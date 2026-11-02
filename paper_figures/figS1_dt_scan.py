"""Fig. S1: iterations to chemical accuracy vs delta t / delta t_MF (QRTE: exact, T1, T2) and
epsilon / delta t_MF (QKUD-T1) for eight systems, in the shared house style.

usage: python paper_figures/figS1_dt_scan.py [table_csv] [out_dir]   (defaults: paper_figures/data/figS1_dt_eps_scan.csv, paper_figures/output)
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, StrMethodFormatter

import figstyle as st

HERE = Path(__file__).resolve().parent
csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "figS1_dt_eps_scan.csv"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "output"
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(csv_path.open()))
for r in rows:
    r["f"] = float(r["dt_scale_factor"])
    r["hit"] = int(float(r["hit_iteration"])) if r["hit_iteration"] not in ("", "None") else None
    r["n"] = int(float(r["n_steps_completed"])); r["dt_mf"] = float(r["dt_mf"])
CASES = ("N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2", "hexatriene_equi_cc_pVTZ_0", "hexatriene_equi_cc_pVTZ_2",
         "H4_near_equi_sto3g_0", "H6_near_equi_sto3g_0", "H6_near_equi_cc_pVTZ_0", "H6_near_equi_cc_pVTZ_2")
TITLES = ("N$_2$/cc-pVTZ, spin 0", "N$_2$/cc-pVTZ, spin 2", "Hexatriene/cc-pVTZ, spin 0", "Hexatriene/cc-pVTZ, spin 2",
          "H$_4$/STO-3G, spin 0", "H$_6$/STO-3G, spin 0", "H$_6$/cc-pVTZ, spin 0", "H$_6$/cc-pVTZ, spin 2")
SERIES = {"exact": (":", True), "trotter1": ("-", False), "trotter2": ("-", False), "qkud": ("-", False)}
KEYMAP = {"exact": "exact", "trotter1": "qrte_fixed", "trotter2": "qrte_mf", "qkud": "qkud"}
assert len(rows) == 8 * 4 * 11, len(rows)

st.apply()
fig, axes = plt.subplots(4, 2, figsize=(st.TEXTWIDTH, 7.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.08, right=0.985, bottom=0.135, top=0.965, hspace=0.42, wspace=0.10)
for index, (ax, cid) in enumerate(zip(axes.flat, CASES)):
    sel = [r for r in rows if r["case_id"] == cid]
    for s, (ls, open_marker) in SERIES.items():
        key = KEYMAP[s]; color = st.COLORS[key]; marker = st.MARKERS[key]
        pr = sorted([r for r in sel if r["series"] == s], key=lambda r: r["f"])
        x = [r["f"] for r in pr]; y = [r["hit"] if r["hit"] is not None else np.nan for r in pr]
        ax.plot(x, y, color=color, ls=ls, lw=1.1, marker=marker, ms=st.MS - (0.6 if s != "exact" else -0.4),
                mfc="white" if open_marker else color, mec=color, mew=0.9 if open_marker else 0.5,
                zorder=4 if s == "exact" else 5)
        for r in pr:
            if r["hit"] is None:
                ax.plot(r["f"], r["n"], ls="none", marker=marker, ms=st.MS - 0.6, mfc="white", mec=color, mew=0.9, zorder=6)
                st.no_hit_arrow(ax, r["f"], r["n"], color)
    ax.axvline(1, color=st.COLORS["guide"], ls=(0, (3, 3)), lw=0.8, zorder=2)
    ax.set(xscale="log", yscale="log", xlim=(0.008, 2.4), ylim=(0.8, 330))
    ax.xaxis.set_major_locator(FixedLocator((0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2)))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator((1, 3, 10, 30, 100)))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    st.grid(ax)
    st.panel_label(ax, chr(97 + index), TITLES[index], y=1.04)
    ax.text(1.0, 1.04, rf"$\delta t_{{\mathrm{{MF}}}}={sel[0]['dt_mf']:.3f}$ a.u.", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#444444")
    if index >= 6:
        ax.tick_params(axis="x", labelrotation=40)
fig.supxlabel(r"$\delta t/\delta t_{\mathrm{MF}}$ (QRTE)   or   $\epsilon/\delta t_{\mathrm{MF}}$ (QKUD-T1)", fontsize=9, y=0.075)
fig.supylabel(st.YLABEL, fontsize=9, x=0.012)
handles = [st.series_handle("exact", st.SHORT_LABELS["exact"], open_marker=True, ls=":", lw=1.1),
           st.series_handle("qrte_fixed", "QRTE-T1", ls="-", lw=1.1),
           st.series_handle("qrte_mf", "QRTE-T2", ls="-", lw=1.1),
           st.series_handle("qkud", r"QKUD-T1 ($\epsilon/\delta t_{\mathrm{MF}}$ on the same axis)", ls="-", lw=1.1),
           Line2D([], [], color=st.COLORS["guide"], ls=(0, (3, 3)), lw=0.8, label=r"$\delta t=\delta t_{\mathrm{MF}}$"),
           st.no_hit_handle()]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, columnspacing=1.2, handlelength=1.8, handletextpad=0.6)
st.save(fig, out / "figS1_dt_eps_scan")
print("ok", len(rows), "rows")
