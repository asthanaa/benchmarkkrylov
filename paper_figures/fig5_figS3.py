"""Fig. 5 (iterations vs Renyi entropy, with ADAPT-VQE) and Fig. S3 (iterations vs
squared Frobenius norm of the two-body cumulant), drawn in the shared house style.

usage: python paper_figures/fig5_figS3.py [repo_root] [out_dir]   (defaults: repo root, paper_figures/output)
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle as st

HERE = Path(__file__).resolve().parent
repo = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "output"
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((repo / "plots/molvqe27_adapt_qrte_dt005_qrte_wmf_qkud_eps005_renyi.csv").open()))
assert len(rows) == 108, len(rows)
for r in rows:
    r["hit"] = str(r["hit_chemical_accuracy"]).lower() == "true"
cum = {r["case_id"]: float(r["cumulant_2rdm_frobenius_norm_squared"])
       for r in csv.DictReader((repo / "plots/qrte_qkud_iterations_vs_2body_cumulant_norm2.csv").open())}
STAR_CASES = {"FeO", "feno_2plus"}   # QKUD-T1 converged, both QRTE protocols did not

st.apply()


def draw(ax, series_keys, xvalue, xlabel, size_scale=1.0):
    r2 = {}
    for key in series_keys:
        pts = [r for r in rows if r["series"] == key and r["hit"]]
        x = np.array([xvalue(r) for r in pts]); y = np.array([float(r["iteration"]) for r in pts])
        xf, yf, r2[key] = st.fit_semilog(x, y)
        shown = [r for r in pts if key != "qkud" or r["case_id"] not in STAR_CASES]
        is_adapt = key == "adapt"
        ax.scatter([xvalue(r) for r in shown], [float(r["iteration"]) for r in shown],
                   marker=st.MARKERS[key], s=(st.SCATTER_S + (6 if is_adapt else 0)) * size_scale, color=st.COLORS[key],
                   alpha=0.45 if is_adapt else 0.9, linewidths=0.5,
                   edgecolors=st.COLORS[key] if is_adapt else "white",
                   label=st.LABELS[key], zorder=2 if is_adapt else 5)
        ax.plot(xf, yf, color=st.COLORS[key], alpha=0.45 if is_adapt else 0.85, ls=":", lw=1.4, zorder=3)
    stars = [r for r in rows if r["series"] == "qkud" and r["case_id"] in STAR_CASES and r["hit"]]
    ax.scatter([xvalue(r) for r in stars], [float(r["iteration"]) for r in stars], marker="*", s=110 * size_scale,
               color=st.COLORS["star"], edgecolors="#333333", linewidths=0.6, zorder=8)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 650)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(st.YLABEL)
    st.grid(ax)
    ax.legend(loc="upper left", handlelength=1.2)
    return r2


# Fig. 5 (two-column figure in the main text: larger type than the SI figures) ---
with plt.rc_context({"font.size": 10.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5,
                     "ytick.labelsize": 10.5, "legend.fontsize": 10.5}):
    fig, ax = plt.subplots(figsize=(st.TEXTWIDTH, 3.9))
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.98)
    r2 = draw(ax, ("adapt", "qrte_fixed", "qrte_mf", "qkud"), lambda r: float(r["entropy_nats"]),
              r"Rényi entropy ($h_{0.25}$)", size_scale=1.35)
    st.save(fig, out / "fig5_molvqe27_renyi")
    plt.close(fig)
print("Fig. 5 R2:", {k: round(v, 3) for k, v in r2.items()})

# Fig. S3 --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(st.TEXTWIDTH, 3.55))
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.135, top=0.98)
r2 = draw(ax, ("qrte_fixed", "qrte_mf", "qkud"), lambda r: cum[r["case_id"]],
          r"$\|{}^{2}\Delta\|_{\mathrm{F}}^{2}$")
st.save(fig, out / "figS3_cumulant")
plt.close(fig)
print("Fig. S3 R2:", {k: round(v, 3) for k, v in r2.items()})
