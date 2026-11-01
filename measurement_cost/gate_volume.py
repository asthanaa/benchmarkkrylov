"""Step 5: two-qubit-gate executions (CNOTs x shots, Fig. S4) of the QRTE-T2 estimate of Fig. 6.

For each system the shot-optimal allocation selected by shot_allocation.py (mc_k, mc_tau, mc_alloc) is rebuilt and
every circuit is charged its CNOT count (Cc, Cu = controlled / uncontrolled T2 step = 2 x T1, from resources.csv):
  H'_jj : j*Cu          1 circuit
  H'_jl : j*Cu + (l-j)*Cc   Re and Im circuits
  S_m   : m*Cc          Re and Im circuits
Shots per circuit are NK * w_c (sum over circuits of w_c = 1), so GK = NK * sum_c w_c * CNOT_c = NK x (mean CNOTs per
shot).  Not counted: the controlled Pauli string that ends each Hadamard test of H' (<= N two-qubit gates per shot)
and single-qubit gates.

usage:  python gate_volume.py [--circuits case_id]
output: results/gate_volume.csv (one row per system); with --circuits, results/circuits_<case>.csv listing every
        circuit (type, j, l, number of circuits, weight w) for a hand check (the workbook uses H4)
"""
import csv
import sys

import numpy as np

import paths
from shot_allocation import solve

rule = "W"
alloc = paths.read_csv(paths.RESULTS / f"alloc_{rule}.csv")


def circuits(z, k, tau, mode_alloc):
    """Circuit list and shot weights of the allocation (same formulas as shot_allocation.analyse)."""
    lam, c_id = float(z["lam"]), float(z["c_id"])
    d = k + 1
    H0 = (z["H"] - c_id * z["S"])[:d, :d]
    S0 = z["S"][:d, :d]
    e0, c = solve(H0, S0, tau)
    a = np.abs(c[:, 0])
    iu = np.triu_indices(d, 1)
    rows = [("H diag", j, j, 1, lam, a[j] ** 2) for j in range(d)]
    rows += [("H off (Re+Im)", int(j), int(l), 2, lam, 2 * a[j] * a[l]) for j, l in zip(*iu)]
    rows += [("S Toeplitz (Re+Im)", 0, m, 2, 1.0, 2 * abs(e0) * np.sum(a[:d - m] * a[m:])) for m in range(1, d)]
    sens = np.array([r[5] for r in rows]); sig = np.array([r[4] for r in rows]); mult = np.array([r[3] for r in rows])
    w = np.ones_like(sens) if mode_alloc == "uniform" else np.maximum(sig * sens, 1e-2 * np.max(sig * sens))
    return rows, w / np.sum(mult * w)


if "--circuits" in sys.argv:
    cid = sys.argv[sys.argv.index("--circuits") + 1]
    a = alloc[cid]
    z = np.load(paths.npz_path(rule, cid))
    rows, w = circuits(z, int(a["mc_k"]), float(a["mc_tau"]), a["mc_alloc"])
    out = paths.RESULTS / f"circuits_{cid}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["type", "j", "l", "circuits", "w"])
        for r, wi in zip(rows, w):
            wr.writerow([r[0], r[1], r[2], r[3], f"{wi:.12e}"])
    print("wrote", out)
    sys.exit()

out = []
for cid, a in sorted(alloc.items()):
    z = np.load(paths.npz_path(rule, cid))
    Cc, Cu = float(z["Cc"]), float(z["Cu"])
    rows, w = circuits(z, int(a["mc_k"]), float(a["mc_tau"]), a["mc_alloc"])
    mult = np.array([r[3] for r in rows])
    g = np.array([r[1] * Cu + (r[2] - r[1]) * Cc for r in rows])
    kind = np.array([r[0][0] for r in rows])
    per_shot = float(np.sum(mult * w * g))
    NK = float(a["mc_shots"])
    out.append({"case_id": cid, "K": int(z["K"]), "vectors": z["S"].shape[0], "k": int(a["mc_k"]), "tau": float(a["mc_tau"]),
                "alloc": a["mc_alloc"], "circuits": int(mult.sum()), "Cc": Cc, "Cu": Cu,
                "shots_NK": NK, "frac_shots_H": float(np.sum((mult * w)[kind == "H"])),
                "cnot_per_shot": per_shot, "ctrl_steps_per_shot": per_shot / Cc, "GK": NK * per_shot,
                "GK_gate_weighted_alloc": float(a["mcg_gateshots"])})
    r = out[-1]
    print(f"{cid[:24]:24s} k={r['k']:3d} circuits={r['circuits']:5d} CNOT/shot={per_shot:.4g} "
          f"({r['ctrl_steps_per_shot']:.1f} controlled steps) GK={r['GK']:.3g}", flush=True)
paths.write_csv(paths.RESULTS / "gate_volume.csv", out)
