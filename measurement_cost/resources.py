"""Step 1: Pauli terms, one-norm lambda, and CNOT counts of one Trotter step for every benchmark system.

The qubit Hamiltonian is the one used by the QRTE-T2 production runs of Fig. 5 (QCANT payload, Jordan-Wigner).
CNOT rule (no cancellation between terms), for a Pauli rotation exp(-i c P t) of weight w:
  uncontrolled  2(w-1)  (CNOT staircase)
  controlled    2w      (only the central Rz is controlled for the Hadamard test)
Terms with |c| <= 1e-10 and the identity are excluded.  A first-order step (T1) has one rotation per term; the
second-order step is counted as 2 x T1 by krylov_bases.py (Cc, Cu); the exact symmetric T2 step (2L-1 rotations,
middle term merged) is listed for comparison (column cnot_T2_step_ctrl_exact, 0.1-0.3% smaller).

usage:  python resources.py [case_id ...]      (default: all systems of Fig. 5)
output: results/resources.csv
"""
import sys

import numpy as np

import paths

CUT = 1e-10
cases = sys.argv[1:] or sorted(paths.fig5_rows())
rows = paths.read_csv(paths.RESULTS / "resources.csv")
h = {c: float(r["entropy_nats"]) for c, r in paths.fig5_rows().items()}
for cid in cases:
    p, _ = paths.payload(cid)
    tw = [(len(t["indices"]), abs(float(np.real(t["coeff"])))) for t in p["pauli_terms"] if len(t["indices"]) > 0]
    w = [x for x, c in tw if c > CUT]
    c_ctrl = [2 * x for x in w]
    c_unc = [2 * (x - 1) for x in w]
    C1, C1u = sum(c_ctrl), sum(c_unc)
    rows[cid] = {"case_id": cid, "h025": h[cid], "n_qubits": p["qubits"], "n_pauli": len(w),
                 "mean_weight": float(np.mean(w)), "lambda": sum(c for _, c in tw),
                 "cnot_T1_step_ctrl": C1, "cnot_T1_step_unctrl": C1u,
                 "cnot_T2_step_ctrl_exact": 2 * C1 - c_ctrl[-1]}
    print(f"{cid:26s} n={p['qubits']:2d} L={len(w):5d} lambda={rows[cid]['lambda']:.4f} T1 ctrl={C1} unctrl={C1u}", flush=True)
paths.RESULTS.mkdir(parents=True, exist_ok=True)
paths.write_csv(paths.RESULTS / "resources.csv", list(rows.values()))
