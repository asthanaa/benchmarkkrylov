#!/usr/bin/env python
"""Exact QRTE scan for eight cases with at most 12 qubits, dt=0.01..2*pi/W_MF.

Eleven factors per case; task indices 0..87. Uses saved Hamiltonians, HF
spectra, initial determinants and CASSCF references, with no Trotterization.
"""

import hashlib
import shutil

import run_qrte_mean_field_active_dt_scan4 as base

qrte, mf = base.qrte, base.mf
CASE_IDS = (
    "N2_equi_cc_pVTZ_0", "N2_equi_cc_pVTZ_2",
    "BeH2_equi_cc_pVTZ_0", "BeH2_equi_cc_pVTZ_2",
    "H4_near_equi_sto3g_0", "H6_near_equi_sto3g_0",
    "H6_near_equi_cc_pVTZ_0", "H6_near_equi_cc_pVTZ_2",
)
FACTORS = (.01, .025, .05, .1, .25, .5, .75, 1., 1.25, 1.5, 2.)
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_dt_scan_upto12_exact"
REUSE_ROOTS = (
    qrte.ROOT / "runs/qrte_mean_field_active_dt_scan4_exact",
    qrte.ROOT / "runs/qrte_mean_field_active_dt_scan_beh2_hchains_exact",
)
run_task = base.run_task


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="sparse")


def load_tasks(args):
    tasks = base.load_tasks(args, case_ids=CASE_IDS, factors=FACTORS)
    if any(2 * task["case"]["active_orbitals"] > 12 for task in tasks):
        raise ValueError("This scan only permits systems with at most 12 qubits")
    return tasks


def reuse_completed(tasks, args):
    """Copy compatible completed runs, preserving originals and recording hashes."""
    import collect_qrte_mean_field_active_dt_scan4 as collector

    reused = 0
    for task in tasks:
        destination = task["outdir"] / task["case"]["case_id"]
        if destination.exists():
            continue
        for root in REUSE_ROOTS:
            source = root / base.factor_label(task["factor"]) / task["case"]["case_id"]
            if not (source / "summary.json").exists():
                continue
            source_task = {**task, "outdir": source.parent}
            result = collector.result_rows([source_task], args)[0]
            if result["status"] not in ("completed_hit", "completed_no_hit"):
                continue
            summary = qrte.load_json(source / "summary.json")
            expected_profile = {
                "fock_profile": {"case": task["case"], "scf_match_tolerance": args.scf_match_tolerance,
                                 "spectrum_scope": base.SPECTRUM_SCOPE},
                "spectrum_scope": base.SPECTRUM_SCOPE, "n_steps": args.n_steps,
                "overlap_tol": args.overlap_tol, "evolution_mode": "sparse", "trotter_steps": 1,
                "max_workers": args.max_workers, "chemical_accuracy_hartree": .0016,
                "qcant_root": str(args.qcant_root.resolve()),
            }
            if summary.get("run_profile") != expected_profile:
                continue
            filenames = ("summary.json", "run_config.json", "input_snapshot.json",
                         "min_energy_history.csv", mf.SPECTRUM_FILENAME)
            if not all((source / name).is_file() for name in filenames):
                continue
            fingerprints = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                            for name in filenames}
            destination.mkdir(parents=True, exist_ok=False)
            for name in filenames:
                shutil.copy2(source / name, destination / name)
            qrte.write_json(destination / "reused_from.json", {
                "source_directory": str(source.resolve()), "copied_file_sha256": fingerprints,
                "validation": "Exact run profile, timestep, Hamiltonian/HF hashes, cached reference and history-derived hit",
                "new_qrte_calculation": False,
            })
            reused += 1
            break
    print(f"Reused {reused} compatible completed exact runs; original results preserved")
    return reused


def main():
    base.main(case_ids=CASE_IDS, factors=FACTORS, default_outdir=DEFAULT_OUTDIR,
              description=__doc__, evolution_mode="sparse", task_loader=load_tasks)


if __name__ == "__main__":
    main()
