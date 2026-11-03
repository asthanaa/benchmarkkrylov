"""Table I of the main text (file name kept from its earlier place as Table S4 of the SI): the STO-3G hydrogen chains H4-H9 from results/depth.csv (step 7), ordered by qubit
count N, with iterations, shots, and CNOT count of the deepest circuit for QRTE-T2 (dt = pi/W) and ADAPT-VQE,
their depth ratio, and power-law fits in N.

usage: python measurement_cost/table_s4_hchains.py      (writes results/table_s4.tex; run circuit_depth.py first)
"""
import csv
import math

import numpy as np

import paths

CHAINS = [("H$_4$", "H4_near_equi_sto3g_0"), ("H$_5$", "H5_near_equi_sto3g_1"), ("H$_6$", "H6_near_equi_sto3g_0"),
          ("H$_7$", "H7_near_equi_sto3g_1"), ("H$_8$", "H8_near_equi_sto3g_0"), ("H$_9$", "H9_near_equi_sto3g_1")]


def sci(x):
    e = int(math.floor(math.log10(x)))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"${m:.1f}\\times10^{{{e}}}$"


def small(x):
    """CNOT counts below 10000 are printed as integers."""
    return f"{x:.0f}" if x < 1e4 else sci(x)


def ratio(x):
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")


M = {r["case_id"]: r for r in csv.DictReader((paths.RESULTS / "depth.csv").open())}
lines, N, NK, DK, NA, DA = [], [], [], [], [], []
for label, cid in CHAINS:
    r = M[cid]
    n, nk, dk, na, da = int(r["N"]), float(r["NK"]), float(r["depth_qrte_fig6"]), float(r["NA"]), float(r["depth_adapt"])
    N.append(n); NK.append(nk); DK.append(dk); NA.append(na); DA.append(da)
    lines.append(f"{label} & {n} & {float(r['h']):.1f} & {float(r['lam']):.1f} & {r['K']} & {r['k_fig6']} & {sci(nk)} & "
                 f"{small(dk)} & {r['KA']} & {sci(na)} & {small(da)} & {ratio(dk / da)} \\\\")


def expo(y):
    return np.polyfit(np.log(N), np.log(y), 1)[0]


fit = (f"Fit & & & & & & $N^{{{expo(NK):.1f}}}$ & $N^{{{expo(DK):.1f}}}$ & & $N^{{{expo(NA):.1f}}}$ & "
       f"$N^{{{expo(DA):.1f}}}$ & \\\\")
body = "\n".join(lines)
tex = rf"""\begin{{tabular}}{{lrrrrrrrrrrr}}
\toprule
 & & & & \multicolumn{{4}}{{c}}{{QRTE-T2, $\delta t=\pi/W$}} & \multicolumn{{3}}{{c}}{{ADAPT-VQE}} & \\
\cmidrule(lr){{5-8}}\cmidrule(lr){{9-11}}
Molecule & $N$ & $h_{{0.25}}$ & $\lambda$ (Ha) & $K$ & $k$ & Shots & CNOTs & $K_A$ & Shots & CNOTs & Ratio \\
\midrule
{body}
\midrule
{fit}
\bottomrule
\end{{tabular}}
"""
out = paths.RESULTS / "table_s4.tex"
out.write_text(tex)
print(tex)
print(f"wrote {out}")
