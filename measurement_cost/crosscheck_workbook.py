"""Step 7 (optional): cross-check workbook with live formulas (results/measurement_crosscheck.xlsx).

Sheets: "Read me" (counting rules), "Summary" (one row per system; inputs in blue, formulas in black, links in green),
"QRTE H4 circuits" (every circuit of the H4 estimate of Fig. 6; totals reproduce NK and GK), "ADAPT iterations"
(all TETRIS-ADAPT-VQE iterations with p_k and c_k; SUMIF gives E_A and G_A).
Needs results/circuits_H4_near_equi_sto3g_0.csv (python gate_volume.py --circuits H4_near_equi_sto3g_0).
Recalculate in Excel/LibreOffice after opening (openpyxl writes formulas without cached values).

usage:  python crosscheck_workbook.py
"""
import csv, json, re
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter as L

import paths

repo = paths.REPO
R = paths.RESULTS
res, gv, ms = paths.read_csv(R / "resources.csv"), paths.read_csv(R / "gate_volume.csv"), paths.read_csv(R / "measurement.csv")
ada = paths.read_csv(R / "adapt_lb.csv")
rc = {c: {"n_qubits": res[c]["n_qubits"], "L_pauli": res[c]["n_pauli"], "mean_w": res[c]["mean_weight"],
          "lambda": res[c]["lambda"], "T1_ctrl": res[c]["cnot_T1_step_ctrl"], "T1_unctrl": res[c]["cnot_T1_step_unctrl"],
          "K_noiseless": gv[c]["K"], "shot_k": gv[c]["k"], "shot_tau": gv[c]["tau"], "shot_alloc": gv[c]["alloc"],
          "NK": gv[c]["shots_NK"], "shot_cnot_per_shot": gv[c]["cnot_per_shot"], "K_A": ada[c]["K_A"],
          "GK_gate_alloc": gv[c]["GK_gate_weighted_alloc"]} for c in ms}
src = open(paths.HERE / "build_figures.py").read()
names = dict(re.findall(r'"([^"]+)": \("([^"]+)", \d+\)', src)); mults = {k: int(v) for k, v in re.findall(r'"([^"]+)": \("[^"]+", (\d+)\)', src)}
man = {r["case_id"]: r for r in csv.DictReader(open(paths.MANIFEST))}
order = sorted(ms, key=lambda c: float(ms[c]["h"]))
latex = lambda s: s.replace("$", "").replace("_{", "").replace("_", "").replace("^{", "").replace("^", "").replace("}", "")

F = "Arial"; BLUE = Font(name=F, color="0000FF"); BLK = Font(name=F); GRN = Font(name=F, color="008000"); BOLD = Font(name=F, bold=True)
HDR = PatternFill("solid", fgColor="DDEBF7")
wb = Workbook()

# ---------------- Read me
rm = wb.active; rm.title = "Read me"
txt = [
    ("Cross-check of Fig. 6, Fig. S4 and Table S3 (QRTE-T2 with dt = pi/W, exact W; ADAPT-VQE lower bound)", BOLD),
    ("", None),
    ("sigma (Ha), target standard deviation of the energy", None),
    ("", None),
    ("Colour code: blue = input data (from the simulations / Hagelueken et al. records); black = formula; green = link to another sheet.", None),
    ("", None),
    ("Shot model (both methods): Pauli terms c_k P_k of H (identity excluded) sampled with probability |c_k|/lambda, lambda = sum|c_k|.", None),
    ("  One energy evaluation with standard deviation sigma costs lambda^2/sigma^2 shots (column 'shots per energy evaluation').", None),
    ("", None),
    ("CNOT rule (QRTE-T2): Pauli rotation of weight w = 2(w-1) CNOTs uncontrolled, 2w controlled (only the central Rz is controlled); no cancellation.", None),
    ("  T1 step = sum over the L Pauli terms; T2 step counted as 2 x T1 (exact symmetric T2 has 2L-1 rotations: 0.1-0.3% fewer CNOTs).", None),
    ("  Circuit for H_jl (j<l): U^j uncontrolled, then controlled U^(l-j): j*Cu + (l-j)*Cc CNOTs; Re and Im parts are 2 circuits.", None),
    ("  Circuit for H_jj: U^j uncontrolled, then measure the sampled Pauli: j*Cu.  Circuit for S_m (Toeplitz): controlled U^m: m*Cc; Re and Im.", None),
    ("  The final controlled Pauli of each H Hadamard test (<= N two-qubit gates) and the HF preparation (X gates) are not counted.", None),
    ("QRTE-T2 shots: Monte Carlo (400 noise samples) over basis dimension k+1 and threshold tau; smallest total shots NK with noiseless energy", None),
    ("  within 1.6 mHa of the reference and energy standard deviation <= sigma. Shots per circuit = NK x w (w = sensitivity weight; sheet 'QRTE H4 circuits').", None),
    ("  Gate executions GK = sum over circuits of shots x CNOTs = NK x (mean CNOTs per shot) for the SAME allocation as Fig. 6.", None),
    ("ADAPT-VQE per iteration k: 4N evaluations for the pool gradients (on the iteration k-1 ansatz) + 1 energy + 2 p_k parameter-shift", None),
    ("  evaluations (on the iteration k ansatz, p_k parameters).  E_A = sum_k (4N + 1 + 2 p_k);  G_A = sum_k [4N cnot_(k-1) + (1 + 2 p_k) cnot_k].", None),
    ("  p_k and cnot_k: TETRIS-ADAPT-VQE records of Hagelueken et al. (zenodo 19051536); K_A = first iteration within 1.6 mHa (noiseless).", None),
    ("  NA = E_A x lambda^2/sigma^2, GA = G_A x lambda^2/sigma^2 (same lambda as QRTE-T2). Lower bound: real optimizers need more evaluations.", None),
    ("Accuracy criterion is the same for both methods: noiseless error <= 1.6 mHa, plus shot noise with standard deviation sigma = 1.6 mHa.", None),
]
for i, (t, f) in enumerate(txt, 1):
    c = rm.cell(row=i, column=1, value=t); c.font = f or BLK
rm["B3"] = 0.0016; rm["B3"].font = BLUE
rm["B3"].comment = Comment("Chemical accuracy, 1.6 mHa (paper's definition).", "measurement_cost")
rm.column_dimensions["A"].width = 120
SIG = "'Read me'!$B$3"

# ---------------- ADAPT iterations
ad = wb.create_sheet("ADAPT iterations")
hdr = ["case_id", "iteration k", "N (qubits)", "p_k (parameters)", "cnot_k", "cnot_(k-1)", "energy evals 4N+1+2p_k",
       "pool CNOT-evals 4N*cnot_(k-1)", "optim. CNOT-evals (1+2p_k)*cnot_k", "CNOT-evals this iteration"]
for j, h in enumerate(hdr, 1):
    c = ad.cell(row=1, column=j, value=h); c.font = BOLD; c.fill = HDR; c.alignment = Alignment(wrap_text=True)
r = 2
for cid in order:
    m = man[cid]; KA = int(m["adapt_chem_accuracy_iteration"]); N = 2 * int(m["active_orbitals"])
    d = json.load(open(repo / m["source_json_path"]))
    sp = sorted([s["attrs"] for s in d["spans"] if s["type"] == "adapt_iter" and s["attrs"]["iter"] < KA], key=lambda a: a["iter"])
    for a in sp:
        vals = [cid, a["iter"] + 1, N, a["num_parameters"], a["num_cnots"]]
        for j, v in enumerate(vals, 1):
            ad.cell(row=r, column=j, value=v).font = BLUE
        ad.cell(row=r, column=6, value=f'=IF(A{r}=A{r-1},E{r-1},0)' if r > 2 else 0).font = BLK
        ad.cell(row=r, column=7, value=f"=4*C{r}+1+2*D{r}").font = BLK
        ad.cell(row=r, column=8, value=f"=4*C{r}*F{r}").font = BLK
        ad.cell(row=r, column=9, value=f"=(1+2*D{r})*E{r}").font = BLK
        ad.cell(row=r, column=10, value=f"=H{r}+I{r}").font = BLK
        r += 1
ad_last = r - 1
for j, w in enumerate([24, 10, 10, 12, 10, 12, 14, 16, 18, 16], 1):
    ad.column_dimensions[L(j)].width = w
ad.freeze_panes = "A2"

# ---------------- Summary
sm = wb.create_sheet("Summary", 1)
cols = [("Molecule", 16), ("Mult.", 6), ("case_id", 24), ("h0.25", 7), ("N qubits", 7), ("Pauli terms L", 8), ("mean weight", 8),
        ("lambda (Ha)", 9), ("T1 step CNOT, controlled", 11), ("T1 step CNOT, uncontrolled", 11), ("Cc = T2 controlled (2xT1)", 11),
        ("Cu = T2 uncontrolled (2xT1)", 11), ("shots per energy evaluation lambda^2/sigma^2", 13), ("K (noiseless iterations)", 9),
        ("k (noisy basis dim - 1)", 9), ("tau", 8), ("allocation", 9), ("QRTE shots NK (Fig. 6)", 12), ("NK in energy evals", 9),
        ("mean CNOTs per shot", 12), ("QRTE gate exec. GK = NK x CNOT/shot (Fig. S4)", 14), ("K_A", 6), ("E_A energy evals", 10),
        ("G_A CNOT-evals", 12), ("ADAPT shots NA = E_A x unit", 12), ("ADAPT gate exec. GA = G_A x unit", 13), ("NA / NK", 9),
        ("GA / GK", 9), ("GK, gate-weighted allocation (not used in the paper)", 14)]
for j, (h, w) in enumerate(cols, 1):
    c = sm.cell(row=1, column=j, value=h); c.font = BOLD; c.fill = HDR; c.alignment = Alignment(wrap_text=True, vertical="top")
    sm.column_dimensions[L(j)].width = w
sm.row_dimensions[1].height = 75
for i, cid in enumerate(order, 2):
    q, mm = rc[cid], ms[cid]
    inp = {1: latex(names[cid]), 2: mults[cid], 3: cid, 4: float(mm["h"]), 5: int(q["n_qubits"]), 6: int(q["L_pauli"]),
           7: float(q["mean_w"]), 8: float(q["lambda"]), 9: int(q["T1_ctrl"]), 10: int(q["T1_unctrl"]), 14: int(q["K_noiseless"]),
           15: int(q["shot_k"]), 16: float(q["shot_tau"]), 17: q["shot_alloc"], 18: float(q["NK"]), 20: float(q["shot_cnot_per_shot"]),
           22: int(q["K_A"]), 29: float(q["GK_gate_alloc"])}
    for j, v in inp.items():
        sm.cell(row=i, column=j, value=v).font = BLUE
    f = {11: f"=2*I{i}", 12: f"=2*J{i}", 13: f"=H{i}^2/{SIG}^2", 19: f"=R{i}/M{i}", 21: f"=R{i}*T{i}",
         23: f"=SUMIF('ADAPT iterations'!$A$2:$A${ad_last},C{i},'ADAPT iterations'!$G$2:$G${ad_last})",
         24: f"=SUMIF('ADAPT iterations'!$A$2:$A${ad_last},C{i},'ADAPT iterations'!$J$2:$J${ad_last})",
         25: f"=W{i}*M{i}", 26: f"=X{i}*M{i}", 27: f"=Y{i}/R{i}", 28: f"=Z{i}/U{i}"}
    for j, v in f.items():
        sm.cell(row=i, column=j, value=v).font = GRN if j in (13, 23, 24) else BLK
    for j in (4, 7, 8):
        sm.cell(row=i, column=j).number_format = "0.00"
    for j in (13, 18, 21, 25, 26, 29):
        sm.cell(row=i, column=j).number_format = "0.00E+00"
    for j in (19, 27, 28):
        sm.cell(row=i, column=j).number_format = "0.0"
    sm.cell(row=i, column=20).number_format = "#,##0"
    sm.cell(row=i, column=16).number_format = "0.0E+00"
sm["R1"].comment = Comment("Smallest total shots from the Monte Carlo search (alloc_mc.py, mode 'shots').", "measurement_cost")
sm["T1"].comment = Comment("sum_c mult_c * w_c * CNOT_c for the Fig. 6 allocation (gate_volume.py); see sheet 'QRTE H4 circuits' for H4.", "measurement_cost")
sm["AC1"].comment = Comment("Previous Fig. S4 values: allocation re-weighted by 1/sqrt(CNOT_c) (alloc_mc.py, mode 'gates'); replaced by column U.", "measurement_cost")
sm.freeze_panes = "D2"

# ---------------- QRTE H4 circuits
hc = wb.create_sheet("QRTE H4 circuits", 2)
h4row = order.index("H4_near_equi_sto3g_0") + 2
hc["A1"] = "H4 (STO-3G), QRTE-T2, dt = pi/W: every circuit of the Fig. 6 estimate (basis dimension 21, tau = 0.316, sensitivity allocation)"; hc["A1"].font = BOLD
lab = [("Cc (controlled T2 step CNOTs)", f"=Summary!K{h4row}"), ("Cu (uncontrolled T2 step CNOTs)", f"=Summary!L{h4row}"),
       ("NK (total shots, Fig. 6)", f"=Summary!R{h4row}")]
for i, (t, v) in enumerate(lab, 2):
    hc.cell(row=i, column=1, value=t).font = BLK; c = hc.cell(row=i, column=2, value=v); c.font = GRN
hc["B4"].number_format = "0.000E+00"
hdr = ["type", "j", "l", "circuits (Re/Im)", "weight w (sum of circuits x w = 1)", "uncontrolled steps j", "controlled steps l-j",
       "CNOTs per circuit", "shots per circuit NK*w", "shots (all circuits of row)", "CNOT x shots"]
R0 = 7
for j, h in enumerate(hdr, 1):
    c = hc.cell(row=R0, column=j, value=h); c.font = BOLD; c.fill = HDR; c.alignment = Alignment(wrap_text=True)
rows = list(csv.DictReader(open(R / "circuits_H4_near_equi_sto3g_0.csv")))
for i, row in enumerate(rows, R0 + 1):
    for j, v in enumerate([row["type"], int(row["j"]), int(row["l"]), int(row["circuits"]), float(row["w"])], 1):
        hc.cell(row=i, column=j, value=v).font = BLUE
    for j, v in {6: f"=B{i}", 7: f"=C{i}-B{i}", 8: f"=F{i}*$B$3+G{i}*$B$2", 9: f"=$B$4*E{i}", 10: f"=D{i}*I{i}", 11: f"=J{i}*H{i}"}.items():
        hc.cell(row=i, column=j, value=v).font = BLK
    hc.cell(row=i, column=5).number_format = "0.000E+00"
    for j in (9, 10, 11):
        hc.cell(row=i, column=j).number_format = "0.000E+00"
last = R0 + len(rows)
tot = [("Total shots (should equal NK)", f"=SUM(J{R0+1}:J{last})", "0.000E+00"),
       ("Total CNOT x shots = GK", f"=SUM(K{R0+1}:K{last})", "0.000E+00"),
       ("Mean CNOTs per shot", f"=B{last+3}/B{last+2}", "#,##0.0"),
       ("Summary sheet GK (check)", f"=Summary!U{h4row}", "0.000E+00"),
       ("Shots in H circuits (fraction)", f'=SUMIF(A{R0+1}:A{last},"H*",J{R0+1}:J{last})/B{last+2}', "0.0%"),
       ("Number of circuits", f"=SUM(D{R0+1}:D{last})", "0")]
for i, (t, v, nf) in enumerate(tot, last + 2):
    hc.cell(row=i, column=1, value=t).font = BOLD; c = hc.cell(row=i, column=2, value=v); c.font = BLK; c.number_format = nf
for j, w in enumerate([34, 16, 6, 9, 14, 11, 11, 11, 13, 13, 13], 1):
    hc.column_dimensions[L(j)].width = w
for ws in wb.worksheets:
    for row in ws.iter_rows():
        for c in row:
            if c.font and c.font.name != F:
                c.font = Font(name=F, bold=c.font.bold, color=c.font.color)
out = R / "measurement_crosscheck.xlsx"
wb.save(out); print(out, ad_last, last)
