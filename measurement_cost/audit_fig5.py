"""Check (optional): krylov_bases.py at dt = pi/W_MF reproduces the production QRTE-T2 runs behind Fig. 5.

Reads the audit JSON files written by `python krylov_bases.py MF ...` (iteration of chemical accuracy and the energy
history compared with each run's min_energy_history.csv).  Result in the paper: 27/27 iteration counts identical,
energies within 1e-8 Ha (up to 1e-5 Ha only in the ill-conditioned late iterations of FeO and FeNO2+).

usage:  python audit_fig5.py
output: results/audit_fig5_WMF.csv
"""
import json

import paths

rows = []
for f in sorted((paths.WORK / "MF").glob("*.audit.json")):
    a = json.load(open(f))
    rows.append({"case_id": a["case_id"], "K_fig5": a["K_fig5"] if a["K_fig5"] else "not reached (100)",
                 "K_rerun": a["K"] if a["K"] else "not reached (%d)" % len(a["hist"]),
                 "iterations_compared": a["n_compared"], "max_abs_dE_Ha": f"{a['max_abs_dE']:.1e}",
                 "match": a["K_fig5"] == a["K"]})
paths.write_csv(paths.RESULTS / "audit_fig5_WMF.csv", rows)
print(len(rows), "systems; all iteration counts match:", all(r["match"] for r in rows),
      "; worst |dE| =", max(float(r["max_abs_dE_Ha"]) for r in rows), "Ha")
