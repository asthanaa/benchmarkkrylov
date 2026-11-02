"""Fig. 3: iterations to chemical accuracy vs (a) squared initial overlap and (b) fixed-sector
gap for the 27-system benchmark (QRTE-T1, QRTE-T2, QKUD-T1; gold stars = QKUD-only hits).

Iteration counts come from the Fig. 5 CSV (the same runs as Table S1); overlap and gap
descriptors from plots/overlap_gap_iterations.csv.
usage: python paper_figures/fig3_overlap_gap.py [repo_root] [out_dir]   (defaults: repo root, paper_figures/output)
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter

import figstyle as st

HERE = Path(__file__).resolve().parent
repo = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "output"
out.mkdir(parents=True, exist_ok=True)
rows = [r for r in csv.DictReader((repo / "plots/molvqe27_adapt_qrte_dt005_qrte_wmf_qkud_eps005_renyi.csv").open())
        if r["series"] != "adapt"]
desc = {}
for r in csv.DictReader((repo / "plots/overlap_gap_iterations.csv").open()):
    desc[r["case_id"]] = (float(r["overlap_weight"]), float(r["gap_ev"]))
cases = sorted({r["case_id"] for r in rows})
assert len(cases) == 27 and all(c in desc for c in cases)
hits = {(r["case_id"], r["series"]): int(r["iteration"]) for r in rows if str(r["hit_chemical_accuracy"]).lower() == "true"}
STAR_CASES = {c for c in cases if ("%s" % c, "qkud") in hits and (c, "qrte_fixed") not in hits and (c, "qrte_mf") not in hits}
assert STAR_CASES == {"FeO", "feno_2plus"}, STAR_CASES

st.apply()
panels = ((0, r"Initial overlap, $|\langle\Phi_{\mathrm{ref}}|\Psi_0\rangle|^2$", "linear", "fig3a_overlap"),
          (1, r"Spectral gap, $E_1-E_0$ (eV)", "log", "fig3b_gap"))
W2 = st.TEXTWIDTH / 2
for col, xlabel, scale, name in panels:
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    fig.subplots_adjust(left=0.165, right=0.965, bottom=0.165, top=0.975)
    for c in cases:   # grey connector through the methods of one system
        ys = [hits[(c, k)] for k in ("qrte_fixed", "qrte_mf", "qkud") if (c, k) in hits]
        if len(ys) > 1:
            ax.plot([desc[c][col]] * 2, [min(ys), max(ys)], color="#b5b5b5", lw=0.6, zorder=1)
    for key in ("qrte_fixed", "qrte_mf", "qkud"):
        pts = [(desc[c][col], hits[(c, key)]) for c in cases if (c, key) in hits and not (key == "qkud" and c in STAR_CASES)]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=st.MARKERS[key], s=st.SCATTER_S,
                   color=st.COLORS[key], alpha=0.9, linewidths=0.5, edgecolors="white", label=st.LABELS[key], zorder=5)
    stars = [(desc[c][col], hits[(c, "qkud")]) for c in sorted(STAR_CASES)]
    ax.scatter([p[0] for p in stars], [p[1] for p in stars], marker="*", s=110, color=st.COLORS["star"],
               edgecolors="#333333", linewidths=0.6, zorder=8)
    ax.set(xscale=scale, yscale="log", xlabel=xlabel)
    st.grid(ax)
    ax.set_ylim(0.8, 150)
    ax.set_ylabel(st.YLABEL)
    if col == 0:
        ax.set_xlim(0.42, 1.0)
        ax.legend(loc="lower left", handlelength=1.2)
    else:
        ax.set_xlim(0.09, 20)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    st.save(fig, out / name)
print("cases", len(cases), "hits", len(hits), "stars", sorted(STAR_CASES))
