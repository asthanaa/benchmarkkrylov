"""Fig. S1 data table: merge the published exact and QRTE-T2 delta-t scans
(plots/qrte_dt_scan_final/qrte_exact_vs_trotter2_loglog.csv) with the QRTE-T1 and QKUD-T1 runs
of figS1_scan_t1_qkud.py (epsilon = f * dt_MF for QKUD) into one CSV for figS1_dt_scan.py.

usage: python paper_figures/figS1_build_table.py [repo_root] [runs_json] [out_csv]
(defaults: repo root, paper_figures/data/figS1_t1_qkud_runs.json, paper_figures/output/figS1_dt_eps_scan.csv)
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
repo = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent
runs_json = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "data" / "figS1_t1_qkud_runs.json"
out_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "output" / "figS1_dt_eps_scan.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

CASES = ("N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2", "hexatriene_equi_cc_pVTZ_0", "hexatriene_equi_cc_pVTZ_2",
         "H4_near_equi_sto3g_0", "H6_near_equi_sto3g_0", "H6_near_equi_cc_pVTZ_0", "H6_near_equi_cc_pVTZ_2")
SERIES = ("exact", "trotter1", "trotter2", "qkud")

pts = {}
for r in csv.DictReader((repo / "plots/qrte_dt_scan_final/qrte_exact_vs_trotter2_loglog.csv").open()):
    hit = int(float(r["hit_iteration"])) if r["status"] == "completed_hit" else None
    pts[(r["case_id"], r["series"], float(r["dt_scale_factor"]))] = (hit, int(float(r["n_steps_completed"])))
    pts.setdefault(("dt_mf", r["case_id"]), float(r["unscaled_mean_field_delta_t"]))
for r in json.loads(runs_json.read_text()):
    pts[(r["case_id"], r["series"], float(r["dt_scale_factor"]))] = (r["hit_iteration"], r["n_steps_completed"])

rows = []
for cid in CASES:
    for s in SERIES:
        for key, val in pts.items():
            if len(key) == 3 and key[0] == cid and key[1] == s:
                hit, n = val
                rows.append({"case_id": cid, "series": s, "dt_scale_factor": key[2], "hit_iteration": hit,
                             "n_steps_completed": n, "dt_mf": pts[("dt_mf", cid)]})
rows.sort(key=lambda r: (CASES.index(r["case_id"]), SERIES.index(r["series"]), r["dt_scale_factor"]))
with out_csv.open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(rows[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out_csv}")
