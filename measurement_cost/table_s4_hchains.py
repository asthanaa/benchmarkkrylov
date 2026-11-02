"""Table S4 of the SI: the STO-3G hydrogen chains H4-H9 from results/measurement.csv, ordered by
qubit count N, with shots, CNOTs per shot (gate executions / shots), gate executions (CNOTs x shots)
for QRTE-T2 (dt = pi/W) and the ADAPT-VQE lower bound, their ratio, and power-law fits in N.

usage: python measurement_cost/table_s4_hchains.py      (writes results/table_s4.tex)
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
    """CNOTs per shot below 1000 are printed as integers."""
    return f"{x:.0f}" if x < 1000 else sci(x)


def ratio(x):
    return f"{x:.2f}" if x < 1 else (f"{x:.1f}" if x < 10 else f"{x:.0f}")


M = {r["case_id"]: r for r in csv.DictReader((paths.RESULTS / "measurement.csv").open())}
lines, N, NK, NA, GK, GA = [], [], [], [], [], []
for label, cid in CHAINS:
    r = M[cid]
    n, nk, na, gk, ga = int(r["N"]), float(r["NK"]), float(r["NA"]), float(r["GK"]), float(r["GA"])
    N.append(n); NK.append(nk); NA.append(na); GK.append(gk); GA.append(ga)
    lines.append(f"{label} & {n} & {float(r['h']):.1f} & {float(r['lam']):.1f} & {int(float(r['K']))} & {sci(nk)} & "
                 f"{small(gk / nk)} & {sci(gk)} & {int(float(r['KA']))} & {sci(na)} & {small(ga / na)} & {sci(ga)} & "
                 f"{ratio(ga / gk)} \\\\")


def expo(y):
    return np.polyfit(np.log(N), np.log(y), 1)[0]


fit = (f"Fit & & & & & $N^{{{expo(NK):.1f}}}$ & & $N^{{{expo(GK):.1f}}}$ & & $N^{{{expo(NA):.1f}}}$ & & "
       f"$N^{{{expo(GA):.1f}}}$ & \\\\")
body = "\n".join(lines)
tex = rf"""\begin{{tabular}}{{lrrrrrrrrrrrr}}
\toprule
 & & & & \multicolumn{{4}}{{c}}{{QRTE-T2, $\delta t=\pi/W$}} & \multicolumn{{4}}{{c}}{{ADAPT-VQE (lower bound)}} & \\
\cmidrule(lr){{5-8}}\cmidrule(lr){{9-12}}
Molecule & $N$ & $h_{{0.25}}$ & $\lambda$ (Ha) & $K$ & Shots & CNOTs/shot & Gate exec. & $K_A$ & Shots & CNOTs/shot & Gate exec. & Ratio \\
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
