"""Step 6: Fig. 6, Fig. S4, Table S3 and results/measurement.csv from the stored results (runs in seconds).

QRTE-T2 uses dt = pi/W (exact sector width) for all systems; the dt = pi/W_MF results (Fig. 5 protocol) are listed in
Table S3 for comparison.
  shots (Fig. 6)          QRTE-T2: NK = mc_shots of results/alloc_W.csv
                          ADAPT-VQE: NA = E_A x lambda^2/sigma^2 (results/adapt_lb.csv)
  gate executions (S4)    QRTE-T2: GK = NK x mean CNOTs per shot of the same allocation (results/gate_volume.csv)
                          ADAPT-VQE: GA = G_A x lambda^2/sigma^2
  lambda, h_0.25, N       results/resources.csv;  W, W_MF: results/widths/
  h_0.25 of the QRTE points is that of the Hamiltonian used for QRTE (resources.csv, as in Fig. 5); the ADAPT-VQE
  points use the dataset value (adapt_lb.csv).  They differ only for the CO triplet (2.578 vs 2.569), whose QRTE runs
  use the MolVQE-matching Hamiltonian.
H10 is omitted because ADAPT-VQE did not reach chemical accuracy in the data of Hagelueken et al.

usage:  python build_figures.py
output: results/fig6_shots.pdf/.png, results/figS4_gates.pdf/.png, results/table_s3.tex, results/measurement.csv
"""
import csv
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

import figstyle as st  # noqa: E402
import paths  # noqa: E402

SIG = paths.SIGMA
out = paths.RESULTS
aw = paths.read_csv(out / "alloc_W.csv")
amf = paths.read_csv(out / "alloc_MF.csv")
ada = paths.read_csv(out / "adapt_lb.csv")
res = paths.read_csv(out / "resources.csv")
gv = paths.read_csv(out / "gate_volume.csv")

rows = []
for cid, a in ada.items():
    if cid not in aw:
        continue
    wj = json.load(open(paths.WIDTHS / f"{cid}.json"))
    lam = float(res[cid]["lambda"]); unit = lam ** 2 / SIG ** 2
    kmf = int(amf[cid]["K"]) if cid in amf else -1
    r = {"case_id": cid, "h": float(res[cid]["h025"]), "h_adapt": float(a["h025"]), "N": int(res[cid]["n_qubits"]), "lam": lam, "unit": unit,
         "W": float(wj["W"]), "W_MF": float(wj["W_MF"]), "K": int(aw[cid]["K"]), "K_MF": kmf if kmf > 0 else None,
         "KA": int(a["K_A"]), "NA": float(a["E_A"]) * unit, "GA": float(a["G_A"]) * unit}
    w = aw[cid]
    r.update(NK=float(w["mc_shots"]), GK=float(gv[cid]["GK"]), cnot_per_shot=float(gv[cid]["cnot_per_shot"]),
             kK=int(w["mc_k"]), tauK=float(w["mc_tau"]), nvec=int(w["vectors"]))
    if cid in amf:
        r.update(NK_MF=float(amf[cid]["mc_shots"]), kK_MF=int(amf[cid]["mc_k"]))
    rows.append(r)
rows.sort(key=lambda r: r["h"])
keys = ["case_id", "h", "h_adapt", "N", "lam", "unit", "W", "W_MF", "K", "kK", "tauK", "nvec", "NK", "cnot_per_shot", "GK",
        "K_MF", "kK_MF", "NK_MF", "KA", "NA", "GA"]
with open(out / "measurement.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)


def fit(pts):
    x = np.array([p[0] for p in pts]); y = np.log([p[1] for p in pts])
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return a, b, r2


st.apply()
fits = {}
for name, kq, ka, ylim, ylabel in (("fig6_shots", "NK", "NA", (1e6, 1e15), "Shots to chemical accuracy"),
                                   ("figS4_gates", "GK", "GA", (1e9, 1e21), r"CNOTs $\times$ shots")):
    fig, ax = plt.subplots(figsize=(st.COLUMNWIDTH, 2.75))
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.155, top=0.97)
    A = [(r["h_adapt"], r[ka]) for r in rows]
    Q = [(r["h"], r[kq]) for r in rows]
    ax.scatter(*zip(*A), marker="o", s=st.SCATTER_S, facecolors="none", edgecolors=st.COLORS["adapt"], linewidths=1.0, zorder=4)
    ax.scatter(*zip(*Q), marker=st.MARKERS["qrte_mf"], s=st.SCATTER_S, facecolors=st.COLORS["qrte_mf"], edgecolors="white",
               linewidths=0.5, alpha=0.9, zorder=5)
    for key, pts in (("adapt", A), ("qrte_mf", Q)):
        a, b, r2 = fit(pts); fits[(name, key)] = (b, r2)
        xx = np.linspace(1.6, 7.0, 50)
        ax.plot(xx, np.exp(a + b * xx), color=st.COLORS[key], ls=":", lw=1.1, zorder=2)
    ax.set_yscale("log"); ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(1.5, 7.0)
    ax.set_xlabel(r"Rényi entropy, $h_{0.25}$"); ax.set_ylabel(ylabel)
    st.grid(ax)
    handles = [Line2D([], [], ls="none", marker="o", ms=st.MS + 0.5, mfc="none", mec=st.COLORS["adapt"], mew=1.0,
                      label="ADAPT-VQE (lower bound)"),
               Line2D([], [], ls="none", marker=st.MARKERS["qrte_mf"], ms=st.MS + 0.5, mfc=st.COLORS["qrte_mf"], mec="white",
                      mew=0.5, label=r"QRTE-T2, $\delta t=\pi/W$"),
               Line2D([], [], color=st.COLORS["guide"], lw=1.1, ls=":", label="Fit")]
    ax.legend(handles=handles, loc="upper left", ncol=1, handlelength=1.5, borderaxespad=0.3, labelspacing=0.3)
    st.save(fig, out / name)
    plt.close(fig)

for k, (b, r2) in fits.items():
    print(k, f"slope={b:.2f}, R2={r2:.2f}")
ratio = [r["NA"] / r["NK"] for r in rows]
print(f"ADAPT/QRTE shot ratio {min(ratio):.3g} .. {max(ratio):.3g}; QRTE below ADAPT for {sum(x > 1 for x in ratio)}/{len(ratio)}")
gr = [r["GA"] / r["GK"] for r in rows]
print("gate ratio > 1 (QRTE fewer gate executions):", [r["case_id"] for r, g in zip(rows, gr) if g > 1])
units = [r["NK"] / r["unit"] for r in rows]
print("QRTE shots in energy evaluations:", sorted(round(u, 1) for u in units))
print("K (exact W):", sorted(r["K"] for r in rows), " K_MF:", [r["K_MF"] for r in rows])
for r in rows:
    print(f"{r['case_id'][:24]:24s} h={r['h']:.2f} lam={r['lam']:5.1f} W={r['W']:.3f} W_MF={r['W_MF']:.3f} K={r['K']:2d} "
          f"K_MF={r['K_MF']} k={r['kK']:3d}/{r['nvec']-1:3d} NK={r['NK']:.2e} ({r['NK']/r['unit']:.1f} ev) NA={r['NA']:.2e} "
          f"ratio={r['NA']/r['NK']:.3g} GK={r['GK']:.2e} GA={r['GA']:.2e} NK_MF={r.get('NK_MF', float('nan')):.2e}")


# Table S3 ------------------------------------------------------------------------------------------------------
NAMES = {
    "H4_near_equi_sto3g_0": ("H$_4$", 1), "cyclobutadiene_ceo": ("C$_4$H$_4$", 1),
    "magnesium_porphyrin_ceo": ("C$_{20}$H$_{12}$MgN$_4$", 1), "H5_near_equi_sto3g_1": ("H$_5$", 2),
    "N2_equi_cc_pVTZ_2": ("N$_2$", 3), "CO_equi_cc_pVTZ_2": ("CO", 3), "N2_equi_cc_pVTZ_0": ("N$_2$", 1),
    "CO_equi_cc_pVTZ_0": ("CO", 1), "BeH2_equi_cc_pVTZ_2": ("BeH$_2$", 3), "BeH2_equi_cc_pVTZ_0": ("BeH$_2$", 1),
    "CH2O_equi_cc_pVTZ_2": ("CH$_2$O", 3), "CH2O_equi_cc_pVTZ_0": ("CH$_2$O", 1),
    "H6_near_equi_cc_pVTZ_2": ("H$_6$ (cc-pVTZ)", 3), "H6_near_equi_cc_pVTZ_0": ("H$_6$ (cc-pVTZ)", 1),
    "H6_near_equi_sto3g_0": ("H$_6$ (STO-3G)", 1), "m_Benzyne_equi_cc_pVTZ_2": ("m-C$_6$H$_4$", 3),
    "m_Benzyne_equi_cc_pVTZ_0": ("m-C$_6$H$_4$", 1), "hexatriene_equi_cc_pVTZ_2": ("C$_6$H$_8$", 3),
    "hexatriene_equi_cc_pVTZ_0": ("C$_6$H$_8$", 1), "H7_near_equi_sto3g_1": ("H$_7$", 2),
    "H8_near_equi_sto3g_0": ("H$_8$", 1), "fenoco3_minus": ("FeNO(CO)$_3^-$", 1), "FeO": ("FeO", 5),
    "Al3_minus": ("Al$_3^-$", 1), "feno_2plus": ("FeNO$^{2+}$", 4), "H9_near_equi_sto3g_1": ("H$_9$", 2),
}


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


lines = []
for r in rows:
    name, mult = NAMES[r["case_id"]]
    kmf = str(r["K_MF"]) if r["K_MF"] else "--"
    nmf = sci(r["NK_MF"]) if "NK_MF" in r else "--"
    lines.append(f"{name} & {mult} & {r['h']:.1f} & {r['lam']:.1f} & {r['K']} & {sci(r['NK'])} & {sci(r['GK'])} & "
                 f"{kmf} & {nmf} & {r['KA']} & {sci(r['NA'])} & {sci(r['GA'])} \\\\")
HEAD = r"""\begin{table}[!htbp]
\centering
\color{red}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.90}
\caption{Measurement cost to chemical accuracy: shots and two-qubit-gate executions (CNOTs $\times$ shots), for Pauli-term sampling with probabilities $|c_k|/\lambda$, $\lambda=\sum_k|c_k|$ (identity excluded), in which one energy evaluation with standard deviation $\sigma=1.6$~mHa requires $\lambda^2/\sigma^2$ shots. QRTE-T2 with $\delta t=\pi/W$, where $W$ is the exact spectral width of the symmetry sector, and, for comparison, with $\delta t=\pi/W_{MF}$ as in Fig.~5 of the main text (``--'': chemical accuracy not reached within 100 iterations). $K$: QRTE-T2 iterations to chemical accuracy. $K_A$: ADAPT-VQE iterations. Gate executions are for the listed shots. ADAPT-VQE values are lower bounds. H$_{10}$ is omitted because ADAPT-VQE did not reach chemical accuracy in the data of Hagelueken \emph{et al.}}
\label{tab:shots}
\begin{tabular}{lrrrrrrrrrrr}
\toprule
 & & & & \multicolumn{3}{c}{QRTE-T2, $\delta t=\pi/W$} & \multicolumn{2}{c}{$\delta t=\pi/W_{MF}$} & \multicolumn{3}{c}{ADAPT-VQE (lower bound)} \\
\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-12}
Molecule & Mult. & $h_{0.25}$ & $\lambda$ (Ha) & $K$ & Shots & Gate exec. & $K$ & Shots & $K_A$ & Shots & Gate exec. \\
\midrule"""
TAIL = r"""\bottomrule
\end{tabular}
\end{table}"""
(out / "table_s3.tex").write_text(HEAD + "\n" + "\n".join(lines) + "\n" + TAIL + "\n")
print("table rows", len(lines))
