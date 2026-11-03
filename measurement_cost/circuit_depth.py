"""Step 7: circuit depth (CNOT count of the deepest circuit), Fig. S4, and Table S3.

Reads results/ only (measurement.csv, gate_volume.csv, adapt_lb.csv).

QRTE-T2: the deepest circuit is the Hadamard test of <ref|U^k|ref>, with k * C_c CNOTs, where C_c is the CNOT
count of one controlled second-order Trotter step (gate_volume.csv) and k is either the iteration of convergence K
(noiseless, 1.6 mHa) or the shot-optimal basis dimension kK of the Fig. 6 estimate.  The counts assume one Pauli
rotation per Hamiltonian term without cancellation or compilation (upper estimates).
ADAPT-VQE: the deepest circuit is the final ansatz; its CNOT count is the compiled count of Hagelueken et al.
(adapt_lb.csv, cnot_final).

output: results/depth.csv, results/figS4_depth.pdf/.png, results/table_s3.tex
usage: python circuit_depth.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

import figstyle as st
import paths

out = paths.RESULTS
M = {r["case_id"]: r for r in csv.DictReader((out / "measurement.csv").open())}
G = {r["case_id"]: r for r in csv.DictReader((out / "gate_volume.csv").open())}
A = {r["case_id"]: r for r in csv.DictReader((out / "adapt_lb.csv").open())}

rows = []
for cid, m in M.items():
    K, k, Cc = int(float(m["K"])), int(float(m["kK"])), float(G[cid]["Cc"])
    rows.append({"case_id": cid, "h": float(m["h"]), "h_adapt": float(m["h_adapt"]), "N": int(m["N"]),
                 "lam": float(m["lam"]), "K": K, "k_fig6": k, "cnot_ctrl_step": Cc,
                 "depth_qrte_K": K * Cc, "depth_qrte_fig6": k * Cc,
                 "KA": int(float(m["KA"])), "depth_adapt": float(A[cid]["cnot_final"]),
                 "NK": float(m["NK"]), "NA": float(m["NA"]), "K_MF": m["K_MF"], "NK_MF": m["NK_MF"]})
rows.sort(key=lambda r: r["h"])
with (out / "depth.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)


def fit(x, y):
    x = np.asarray(x); y = np.log(y); b, a = np.polyfit(x, y, 1)
    return a, b, 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)


# Fig. S4 ------------------------------------------------------------------------------------------------------
st.apply()
green, gray = st.COLORS["qrte_mf"], st.COLORS["adapt"]
fig, ax = plt.subplots(figsize=(st.COLUMNWIDTH, 2.75))
fig.subplots_adjust(left=0.17, right=0.97, bottom=0.155, top=0.97)
for r in rows:
    ax.plot([r["h"], r["h"]], [r["depth_qrte_K"], r["depth_qrte_fig6"]], color=green, lw=0.6, alpha=0.5, zorder=3)
ax.scatter([r["h_adapt"] for r in rows], [r["depth_adapt"] for r in rows], marker="o", s=st.SCATTER_S,
           facecolors="none", edgecolors=gray, linewidths=1.0, zorder=4)
ax.scatter([r["h"] for r in rows], [r["depth_qrte_K"] for r in rows], marker="D", s=st.SCATTER_S * 0.8,
           facecolors="white", edgecolors=green, linewidths=1.0, zorder=5)
ax.scatter([r["h"] for r in rows], [r["depth_qrte_fig6"] for r in rows], marker="D", s=st.SCATTER_S,
           facecolors=green, edgecolors="white", linewidths=0.5, alpha=0.9, zorder=6)
fits = {}
xx = np.linspace(1.6, 7.0, 50)
for key, xs, ys, col, ls, alpha in (
        ("ADAPT final ansatz", [r["h_adapt"] for r in rows], [r["depth_adapt"] for r in rows], gray, ":", 1.0),
        ("QRTE-T2 at K", [r["h"] for r in rows], [r["depth_qrte_K"] for r in rows], green, (0, (1, 3)), 0.7),
        ("QRTE-T2 Fig. 6 basis", [r["h"] for r in rows], [r["depth_qrte_fig6"] for r in rows], green, ":", 1.0)):
    a, b, r2 = fit(xs, ys); fits[key] = (b, r2)
    ax.plot(xx, np.exp(a + b * xx), color=col, ls=ls, lw=1.0, zorder=2, alpha=alpha)
ax.set_yscale("log"); ax.set_ylim(10, 1e8); ax.set_xlim(1.5, 7.0)
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=10))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10), numticks=10))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Rényi entropy, $h_{0.25}$"); ax.set_ylabel("CNOTs in the deepest circuit")
st.grid(ax)
handles = [Line2D([], [], ls="none", marker="o", ms=st.MS + 0.5, mfc="none", mec=gray, mew=1.0, label="ADAPT-VQE, final ansatz"),
           Line2D([], [], ls="none", marker="D", ms=st.MS, mfc="white", mec=green, mew=1.0, label=r"QRTE-T2, converged ($K$)"),
           Line2D([], [], ls="none", marker="D", ms=st.MS + 0.5, mfc=green, mec="white", mew=0.5, label="QRTE-T2, Fig. 6 basis"),
           Line2D([], [], color=st.COLORS["guide"], lw=1.0, ls=":", label="Fit")]
ax.legend(handles=handles, loc="lower right", ncol=1, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2,
          labelspacing=0.25, fontsize=6.5)
st.save(fig, out / "figS4_depth")
plt.close(fig)
for k_, (b, r2) in fits.items():
    print(f"{k_}: exp({b:.2f} h), R2={r2:.2f}")
rk = [r["depth_qrte_K"] / r["depth_adapt"] for r in rows]; rf = [r["depth_qrte_fig6"] / r["depth_adapt"] for r in rows]
print(f"QRTE/ADAPT depth ratio: at K {min(rk):.0f}-{max(rk):.0f} (median {np.median(rk):.0f}); "
      f"Fig. 6 basis {min(rf):.0f}-{max(rf):.0f} (median {np.median(rf):.0f})")
print(f"ADAPT depth {min(r['depth_adapt'] for r in rows):.0f}-{max(r['depth_adapt'] for r in rows):.0f}; "
      f"QRTE at K {min(r['depth_qrte_K'] for r in rows):.2e}-{max(r['depth_qrte_K'] for r in rows):.2e}; "
      f"QRTE Fig. 6 {min(r['depth_qrte_fig6'] for r in rows):.2e}-{max(r['depth_qrte_fig6'] for r in rows):.2e}")

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


def cnots(x):
    return f"{x:.0f}" if x < 1e4 else sci(x)


lines = []
for r in rows:
    name, mult = NAMES[r["case_id"]]
    kmf = r["K_MF"] if r["K_MF"] else "--"
    nmf = sci(float(r["NK_MF"])) if r["NK_MF"] and r["NK_MF"] != "nan" else "--"
    lines.append(f"{name} & {mult} & {r['h']:.1f} & {r['lam']:.1f} & {r['K']} & {r['k_fig6']} & {sci(r['NK'])} & "
                 f"{cnots(r['depth_qrte_fig6'])} & {kmf} & {nmf} & {r['KA']} & {sci(r['NA'])} & {cnots(r['depth_adapt'])} \\\\")
HEAD = r"""\begin{table}[!htbp]
\centering
\color{red}
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{0.90}
\caption{Measurement cost to chemical accuracy, shots and CNOT count of the deepest circuit, for Pauli-term sampling with probabilities $|c_k|/\lambda$, $\lambda=\sum_k|c_k|$ (identity excluded), in which one energy evaluation with standard deviation $\sigma=1.6$~mHa requires $\lambda^2/\sigma^2$ shots. QRTE-T2 with $\delta t=\pi/W$, where $W$ is the exact spectral width of the symmetry sector, and, for comparison, with $\delta t=\pi/W_{MF}$ as in Fig.~5 of the main text (``--'': chemical accuracy not reached within 100 iterations). $K$: QRTE-T2 iterations to chemical accuracy; $k$: Trotter steps of the shot-optimal basis of Fig.~6 of the main text. $K_A$: ADAPT-VQE iterations. CNOTs: deepest circuit of the Fig.~6 estimate (QRTE-T2, Hadamard test of $U^k$; uncompiled upper estimate) or final ansatz (ADAPT-VQE, compiled count of Hagelueken \emph{et al.}). ADAPT-VQE shots are lower bounds. H$_{10}$ is omitted because ADAPT-VQE did not reach chemical accuracy in the data of Hagelueken \emph{et al.}}
\label{tab:shots}
\begin{tabular}{lrrrrrrrrrrrr}
\toprule
 & & & & \multicolumn{4}{c}{QRTE-T2, $\delta t=\pi/W$} & \multicolumn{2}{c}{$\delta t=\pi/W_{MF}$} & \multicolumn{3}{c}{ADAPT-VQE} \\
\cmidrule(lr){5-8}\cmidrule(lr){9-10}\cmidrule(lr){11-13}
Molecule & Mult. & $h_{0.25}$ & $\lambda$ (Ha) & $K$ & $k$ & Shots & CNOTs & $K$ & Shots & $K_A$ & Shots & CNOTs \\
\midrule"""
TAIL = r"""\bottomrule
\end{tabular}
\end{table}"""
(out / "table_s3.tex").write_text(HEAD + "\n" + "\n".join(lines) + "\n" + TAIL + "\n")
print("table rows", len(lines))
