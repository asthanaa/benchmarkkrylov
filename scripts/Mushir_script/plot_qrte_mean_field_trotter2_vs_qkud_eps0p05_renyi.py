#!/home/mushir.kolathingal/miniforge3/envs/expk/bin/python
"""Remake the 27-system Renyi plot using saved mean-field-timestep QRTE.

QRTE uses dt=pi/W_MF, second-order Trotter, and one slice. The original QKUD
eps=0.05 points and ADAPT background are retained. No calculations are launched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import plot_qkud_eps0p05_vs_qrte_dt0p05 as original

ROOT = original.BENCHMARK_ROOT
DEFAULT_PREFIX = ROOT / "plots/qkud_eps0p05_vs_qrte_mean_field_pi_over_w_trotter2_steps1_iterations_vs_entropy"
QRTE_ROOTS = (
    ROOT / "runs/qrte_mean_field_active_pi_over_w_small10_trotter2",
    ROOT / "runs/qrte_mean_field_active_pi_over_w_remaining17_trotter2",
)
ITERATION_KEY = "qrte_mean_field_pi_over_w_chem_accuracy_iteration"
FIELDS = [field.replace("qrte_dt0p05_chem_accuracy_iteration", ITERATION_KEY)
          for field in original.FIELDNAMES] + [
    "qrte_delta_t", "qrte_mean_field_active_bandwidth_hartree", "qrte_trotter_order",
    "qrte_trotter_steps", "qrte_final_error_vs_local_reference_hartree",
    "qrte_local_reference_energy_hartree", "qrte_hamiltonian_sha256",
    "qrte_source_mean_field_reference_sha256", "qkud_epsilon", "qkud_trotter_order",
    "qkud_trotter_steps", "qkud_order_provenance", "qkud_reference_energy_hartree",
    "qkud_reference_minus_qrte_reference_hartree",
]


def load_json(path):
    return json.loads(path.read_text())


def validate_qrte(path, cache_row):
    summary = load_json(path)
    config = load_json(path.parent / "run_config.json")
    cid = cache_row["case_id"]
    if summary.get("case_id") != cid or summary.get("status") != "completed":
        raise ValueError(f"{cid}: missing completed QRTE result")
    for result in (summary, config):
        if (result.get("evolution_mode"), result.get("trotter_order"), result.get("trotter_steps")) != ("trotter", 2, 1):
            raise ValueError(f"{cid}: expected second-order QRTE with one slice")
        width, dt = result["mean_field_active_bandwidth_hartree"], result["delta_t"]
        if not math.isfinite(width) or width <= 0 or not math.isclose(dt, math.pi / width, rel_tol=1e-12):
            raise ValueError(f"{cid}: timestep does not equal pi/W_MF")
        if result.get("spectrum_scope") != "saved_molecular_hf_active_window_fixed_nalpha_nbeta":
            raise ValueError(f"{cid}: wrong mean-field bandwidth definition")
    if summary["hamiltonian_sha256"] != cache_row["sha256"]:
        raise ValueError(f"{cid}: QRTE uses a different Hamiltonian")
    cache_path = ROOT / cache_row["consolidated_cache_path"]
    if hashlib.sha256(cache_path.read_bytes()).hexdigest() != cache_row["sha256"]:
        raise ValueError(f"{cid}: validated Hamiltonian checksum changed")
    hf_path = Path(summary["source_mean_field_reference_path"])
    if hashlib.sha256(hf_path.read_bytes()).hexdigest() != summary["source_mean_field_reference_sha256"]:
        raise ValueError(f"{cid}: source HF spectrum changed")
    with (path.parent / "min_energy_history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    reference = summary["local_reference_energy_hartree"]
    hit = next((int(row["iteration"]) for row in history
                if abs(float(row["min_energy_hartree"]) - reference) <= 0.0016), None)
    if hit != summary["hit_iteration"] or (hit is not None) != summary["hit_chemical_accuracy"]:
        raise ValueError(f"{cid}: QRTE hit flag disagrees with the saved history")
    return summary


def build_rows(manifest_path, cache_manifest):
    manifest = {row["case_id"]: row for row in original._read_manifest(manifest_path)}
    cache_rows = original._read_manifest(cache_manifest)
    if len(cache_rows) != 27 or len({row["case_id"] for row in cache_rows}) != 27:
        raise ValueError("Expected 27 unique validated systems")
    rows = []
    for cached in cache_rows:
        cid = cached["case_id"]
        molecular = manifest[cid]
        paths = [root / cid / "summary.json" for root in QRTE_ROOTS
                 if (root / cid / "summary.json").exists()]
        if len(paths) != 1:
            raise ValueError(f"{cid}: expected exactly one QRTE result in the selected roots")
        qrte_path = paths[0]
        qrte = validate_qrte(qrte_path, cached)
        qkud_path, source, qkud = original._summary_for_case(cid, original.BASE_QKUD_DIR, original.QKUD_RERUN_SUMMARIES)
        if qkud_path.resolve() != (ROOT / cached["summary_path"]).resolve():
            raise ValueError(f"{cid}: QKUD source differs from the validated earlier plot")
        if qkud is None or qkud.get("status") != "completed":
            raise ValueError(f"{cid}: missing saved QKUD result")
        qconfig = load_json(qkud_path.parent / "run_config.json")
        if (qconfig.get("evolution_mode"), qconfig.get("trotter_steps"), qconfig.get("epsilon")) != ("trotter", 1, 0.05):
            raise ValueError(f"{cid}: expected QKUD eps=0.05 and one Trotter slice")
        if qconfig.get("trotter_order", 1) != 1:
            raise ValueError(f"{cid}: QKUD was not first order")
        adapt_iteration, adapt_hit, adapt_source, adapt_note = original._adapt_iteration(molecular)
        qhit, rhit = original._summary_hit(qkud), original._summary_hit(qrte)
        qreference = qkud.get("local_reference_energy_hartree")
        if qreference is None:
            qreference = qkud.get("casscf_energy_hartree", qkud.get("dataset_casscf_energy_hartree"))
        if qreference is None:
            raise ValueError(f"{cid}: missing saved QKUD accuracy reference")
        rows.append({
            "case_id": cid, "source_name": molecular["source_name"], "h0.25": float(molecular["h0.25"]),
            "adapt_chem_accuracy_iteration": adapt_iteration, "adapt_hit_chemical_accuracy": str(adapt_hit).lower(),
            "adapt_source": adapt_source, "adapt_note": adapt_note,
            "qkud_eps0p05_chem_accuracy_iteration": original._iteration(qkud),
            "qkud_hit_chemical_accuracy": str(qhit).lower(), "qkud_status": qkud["status"],
            "qkud_n_steps_completed": original._steps_completed(qkud),
            "qkud_n_steps_requested": original._steps_requested(qkud),
            "qkud_summary_path": original._summary_path_text(qkud_path), "qkud_result_source": source,
            ITERATION_KEY: original._iteration(qrte), "qrte_hit_chemical_accuracy": str(rhit).lower(),
            "qrte_status": qrte["status"], "qrte_n_steps_completed": original._steps_completed(qrte),
            "qrte_n_steps_requested": original._steps_requested(qrte),
            "qrte_summary_path": original._summary_path_text(qrte_path),
            "qrte_result_source": qrte_path.parent.parent.name, "qkud_only_hit": str(qhit and not rhit).lower(),
            "casscf_energy_hartree": float(molecular["casscf_energy_hartree"]),
            "qrte_delta_t": qrte["delta_t"],
            "qrte_mean_field_active_bandwidth_hartree": qrte["mean_field_active_bandwidth_hartree"],
            "qrte_trotter_order": 2, "qrte_trotter_steps": 1,
            "qrte_final_error_vs_local_reference_hartree": qrte["final_error_vs_local_reference_hartree"],
            "qrte_local_reference_energy_hartree": qrte["local_reference_energy_hartree"],
            "qrte_hamiltonian_sha256": qrte["hamiltonian_sha256"],
            "qrte_source_mean_field_reference_sha256": qrte["source_mean_field_reference_sha256"],
            "qkud_epsilon": 0.05, "qkud_trotter_order": 1, "qkud_trotter_steps": 1,
            "qkud_order_provenance": "QKUD circuit builder uses default trotter_order=1; saved config omits order",
            "qkud_reference_energy_hartree": qreference,
            "qkud_reference_minus_qrte_reference_hartree": qreference - qrte["local_reference_energy_hartree"],
        })
    return sorted(rows, key=lambda row: (row["h0.25"], row["case_id"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=original.DEFAULT_MANIFEST)
    parser.add_argument("--cache-manifest", type=Path, default=ROOT / "hamiltonian_cache/correct_27_systems/manifest.csv")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    args = parser.parse_args()
    rows = build_rows(args.manifest, args.cache_manifest)
    qhits = sum(row["qkud_hit_chemical_accuracy"] == "true" for row in rows)
    rhits = sum(row["qrte_hit_chemical_accuracy"] == "true" for row in rows)
    misses = [row["case_id"] for row in rows if row["qrte_hit_chemical_accuracy"] != "true"]
    if (qhits, rhits) != (27, 25):
        raise ValueError(f"Expected 27 QKUD hits and 25 QRTE hits, found {qhits}, {rhits}")
    if set(misses) != {"FeO", "feno_2plus"}:
        raise ValueError(f"Expected FeO and FeNO2+ as the two QRTE misses, found {misses}")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    original._plot(
        rows, args.output_prefix.with_suffix(".png"), args.output_prefix.with_suffix(".pdf"),
        qrte_iteration_key=ITERATION_KEY,
        subtitle=r"QRTE: $\Delta t=\pi/W_{\mathrm{MF}}$, Trotter order 2, one slice"
                 "\n" r"QKUD: $\epsilon=0.05$, Trotter order 1, one slice",
        note="QRTE: 25/27 hits; QKUD: 27/27 hits\nStars: FeO and FeNO²⁺; QRTE did not hit by 100 iterations")
    audit = {
        "case_count": len(rows), "qrte_hits": rhits, "qkud_hits": qhits, "qrte_no_hit_cases": misses,
        "new_molecular_calculations": False,
        "entropy_source": str(args.manifest),
        "entropy_definition": "Original manifest h0.25 from dataset CI probabilities, natural logarithm; unchanged",
        "qrte_roots": [str(root) for root in QRTE_ROOTS],
        "qkud_propagation": "first-order product formula, one slice, epsilon=0.05",
        "qkud_order_evidence": "QCANT.qulacs_accel.qkud_qulacs calls _build_time_evolution_circuit without order; its default is 1",
        "fit_definition": "Least squares of log(iterations) against entropy, successful cases only; R2 in log space",
        "fit_r_squared_log_space": {
            method: original._fit_semilog(*original._finite_xy(rows, key))[2]
            for method, key in (("qrte", ITERATION_KEY),
                                ("qkud", "qkud_eps0p05_chem_accuracy_iteration"),
                                ("adapt", "adapt_chem_accuracy_iteration"))
        },
        "notes": ["Original QKUD points and ADAPT overrides retained; no-hit QRTE cases are excluded from QRTE fit.",
                  "QKUD-only hits are yellow stars; QRTE no-hit rows retain blank hit iterations in CSV.",
                  "Legacy QKUD accuracy targets are unchanged, including CO spin 2 and Al3-minus; reference differences are recorded in CSV.",
                  "fenoco3_minus retains its explicitly recorded unconverged CASSCF/natorb cache."],
    }
    args.output_prefix.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Plotted {rhits}/27 QRTE hits, {qhits}/27 unchanged QKUD hits, and the original ADAPT background.")
    print(f"QRTE no hit by 100 iterations: {', '.join(misses)}")
    for extension in ("png", "pdf", "csv", "json"):
        print(args.output_prefix.with_suffix("." + extension))


if __name__ == "__main__":
    main()
