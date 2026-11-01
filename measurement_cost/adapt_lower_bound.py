"""Step 4: lower bound on the ADAPT-VQE measurement cost to chemical accuracy, from the per-iteration
TETRIS-ADAPT-VQE records of Hagelueken et al. (Zenodo 19051536, MolVQE-21; dataset/ in this repository).

Per ADAPT iteration k = 1..K_A (K_A = first iteration within 1.6 mHa of the CASCI energy, noiseless):
  pool gradients   4N energy-evaluation equivalents, measured in commuting groups (Anastasiou et al.,
                   arXiv:2306.03227), on the ansatz of iteration k-1 (CNOT count c_(k-1), c_0 = 0)
  optimization     1 energy + 2 p_k parameter-shift evaluations on the ansatz of iteration k (p_k parameters, c_k CNOTs)
  E_A = sum_k (4N + 1 + 2 p_k)                        energy evaluations
  G_A = sum_k [4N c_(k-1) + (1 + 2 p_k) c_k]          CNOT-weighted evaluations
Shots are E_A x lambda^2/sigma^2 and gate executions G_A x lambda^2/sigma^2, with the same lambda as QRTE-T2
(build_figures.py).  Real optimizers need more evaluations, so these are lower bounds.

usage:  python adapt_lower_bound.py
output: results/adapt_lb.csv
"""
import csv
import json

import paths

rows = []
for m in csv.DictReader(open(paths.MANIFEST)):
    if not m["adapt_chem_accuracy_iteration"]:
        continue
    d = json.load(open(paths.REPO / m["source_json_path"]))
    KA = int(m["adapt_chem_accuracy_iteration"])
    N = 2 * int(m["active_orbitals"])
    sp = sorted([s["attrs"] for s in d["spans"] if s["type"] == "adapt_iter" and s["attrs"]["iter"] < KA],
                key=lambda a: a["iter"])          # records are 0-based: iterations 0..K_A-1
    assert len(sp) == KA
    E_A, G_A, G_pool, prev = 0.0, 0.0, 0.0, 0
    for a in sp:
        p, cn = a["num_parameters"], a["num_cnots"]
        E_A += 4 * N + 1 + 2 * p
        G_A += 4 * N * prev + (1 + 2 * p) * cn
        G_pool += 4 * N * prev
        prev = cn
    rows.append({"case_id": m["case_id"], "h025": float(m["h0.25"]), "N": N, "K_A": KA,
                 "ansatz": d["meta"].get("ansatz", {}).get("type", ""),
                 "p_final": sp[-1]["num_parameters"], "cnot_final": sp[-1]["num_cnots"],
                 "E_A": E_A, "E_A_pool": 4 * N * KA, "E_A_opt": E_A - 4 * N * KA, "G_A": G_A,
                 "G_A_pool": G_pool, "G_A_opt": G_A - G_pool})
    r = rows[-1]
    print(f"{r['case_id'][:24]:24s} h={r['h025']:.2f} N={N:2d} K_A={KA:3d} E_A={E_A:9.0f} G_A={G_A:.4g}")
paths.RESULTS.mkdir(parents=True, exist_ok=True)
paths.write_csv(paths.RESULTS / "adapt_lb.csv", rows)
