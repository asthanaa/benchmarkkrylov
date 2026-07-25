# LaTeX table for overlap_gap_iterations.csv

Preferred compact layout: one row per molecular system, with QKUD and QRTE iterations in separate columns. Rows are ordered according to `manifests/molvqe21_fig1b_manifest.csv`, filtered to the cases present in `plots/overlap_gap_iterations.csv`. The molecule labels use chemical formulas instead of case IDs, and the source filename column has been removed.

Values below are copied from `plots/overlap_gap_iterations.csv`, `manifests/molvqe21_fig1b_manifest.csv`, `runs/two_body_cumulant_norms/qkud_eps0p05_converged/two_body_cumulant_norms.csv`, and the corresponding run `input_snapshot.json` files for the CAS spaces and spins. Multiplicity is reported as $2S+1$ from the stored spin value $2S$. Numeric quantities are rounded to two significant digits for presentation. Missing QRTE entries are shown as `--` where `plots/overlap_gap_iterations.csv` has no QRTE row for that case.

```latex
% Requires: \usepackage{booktabs,adjustbox}
\begin{table}[htbp]
\centering
\setlength{\tabcolsep}{2pt}
\caption{Overlap, initial-state gap, iterations to chemical accuracy, and two-body cumulant norm squared for the benchmark molecular systems.}
\label{tab:overlap-gap-cumulant}
\begin{tabular}{lllrrrrrrrl}
\toprule
Molecule & Basis & CAS ($n_e,n_o$) & Mult. & QKUD iter. & QRTE iter. & Overlap & Gap (eV) & $h_{0.25}$ & $\|\Delta_2\|_F^2$ & QKUD-only \\
\midrule
H$_4$ & STO-3G & (4,4) & 1 & 3 & 5 & 0.94 & 6.3 & 1.8 & 0.54 & \texttt{false} \\
C$_4$H$_4$ & cc-pVDZ & (4,4) & 1 & 3 & 7 & 0.86 & 1.8 & 1.9 & 1.3 & \texttt{false} \\
C$_{20}$H$_{12}$MgN$_4$ & def2-TZVP & (4,4) & 1 & 2 & 2 & 0.93 & 2.8 & 1.9 & 0.64 & \texttt{false} \\
H$_5$ & STO-3G & (5,5) & 2 & 3 & 4 & 0.92 & 5.9 & 2.7 & 0.50 & \texttt{false} \\
N$_2$ & cc-pVTZ & (6,6) & 3 & 6 & 56 & 0.47 & 1.5 & 2.4 & 3.6 & \texttt{false} \\
CO & cc-pVTZ & (6,6) & 3 & 1 & 1 & 0.94 & 3.6 & 2.6 & 0.44 & \texttt{false} \\
N$_2$ & cc-pVTZ & (6,6) & 1 & 4 & 7 & 0.91 & 7.2 & 3.4 & 0.80 & \texttt{false} \\
CO & cc-pVTZ & (6,6) & 1 & 3 & 3 & 0.94 & 11 & 3.6 & 0.51 & \texttt{false} \\
BeH$_2$ & cc-pVTZ & (4,6) & 3 & 2 & 2 & 0.94 & 0.98 & 1.8 & 0.46 & \texttt{false} \\
BeH$_2$ & cc-pVTZ & (4,6) & 1 & 1 & 1 & 0.98 & 11 & 2.0 & 0.18 & \texttt{false} \\
CH$_2$O & cc-pVTZ & (6,6) & 3 & 2 & 2 & 0.98 & 2.2 & 2.0 & 0.15 & \texttt{false} \\
CH$_2$O & cc-pVTZ & (6,6) & 1 & 3 & 12 & 0.95 & 8.9 & 2.8 & 0.41 & \texttt{false} \\
H$_6$ & cc-pVTZ & (6,6) & 3 & 3 & 5 & 0.93 & 4.2 & 3.3 & 0.53 & \texttt{false} \\
H$_6$ & cc-pVTZ & (6,6) & 1 & 3 & 10 & 0.92 & 4.5 & 3.6 & 0.72 & \texttt{false} \\
H$_6$ & STO-3G & (6,6) & 1 & 4 & 9 & 0.90 & 4.7 & 3.6 & 0.83 & \texttt{false} \\
m-C$_6$H$_4$ & cc-pVTZ & (6,6) & 3 & 3 & 2 & 0.89 & 0.69 & 3.4 & 0.81 & \texttt{false} \\
m-C$_6$H$_4$ & cc-pVTZ & (6,6) & 1 & 3 & 4 & 0.91 & 4.7 & 3.7 & 0.79 & \texttt{false} \\
C$_6$H$_8$ & cc-pVTZ & (6,6) & 3 & 3 & 8 & 0.82 & 1.7 & 3.6 & 1.5 & \texttt{false} \\
C$_6$H$_8$ & cc-pVTZ & (6,6) & 1 & 3 & 6 & 0.87 & 3.3 & 3.9 & 1.1 & \texttt{false} \\
H$_7$ & STO-3G & (7,7) & 2 & 4 & 9 & 0.88 & 4.6 & 4.7 & 0.77 & \texttt{false} \\
H$_8$ & STO-3G & (8,8) & 1 & 5 & 16 & 0.87 & 3.8 & 5.7 & 1.1 & \texttt{false} \\
FeNO(CO)$_3^-$ & def2-TZVP & (10,8) & 1 & 6 & 62 & 0.79 & 3.0 & 5.2 & 2.1 & \texttt{false} \\
FeO & def2-TZVP & (12,9) & 5 & 5 & -- & 0.79 & 0.13 & 3.9 & 1.1 & \texttt{true} \\
Al$_3^-$ & def2-TZVP & (8,9) & 1 & 4 & 16 & 0.82 & 1.6 & 6.4 & 1.6 & \texttt{false} \\
FeNO$^{2+}$ & cc-pVTZ & (11,9) & 4 & 12 & -- & 0.62 & 0.99 & 5.5 & 1.9 & \texttt{true} \\
H$_9$ & STO-3G & (9,9) & 2 & 5 & 15 & 0.85 & 3.8 & 6.7 & 1.0 & \texttt{false} \\
H$_{10}$ & STO-3G & (10,10) & 1 & 6 & 22 & 0.84 & 3.1 & 7.8 & 1.4 & \texttt{false} \\
\bottomrule
\end{tabular}
\end{table}
```
