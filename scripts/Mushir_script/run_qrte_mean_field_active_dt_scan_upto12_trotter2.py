#!/usr/bin/env python
"""Repeat the exact <=12-qubit timestep grid with order-2 Trotter, one slice.

Eight cases, eleven factors, task indices 0..87. Validate every condition
against the saved exact scan before propagating the cached Hamiltonian.
"""

import run_qrte_mean_field_active_dt_scan_upto12_exact as exact

base, qrte, mf = exact.base, exact.qrte, exact.mf
CASE_IDS, FACTORS = exact.CASE_IDS, exact.FACTORS
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_dt_scan_upto12_trotter2"
EXACT_PLAN = exact.DEFAULT_OUTDIR / "scan_plan.json"
run_task = base.run_task
MATCH_FIELDS = (
    "task_index", "case_id", "dt_scale_factor", "delta_t",
    "mean_field_active_bandwidth_hartree", "unscaled_mean_field_delta_t",
    "cached_reference_energy_hartree", "dataset_casscf_energy_hartree",
    "hamiltonian_sha256", "mean_field_reference_sha256",
)


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="trotter")


def load_tasks(args):
    tasks = base.load_tasks(args, case_ids=CASE_IDS, factors=FACTORS)
    validate_exact_match(tasks, args, exact_plan=EXACT_PLAN, case_ids=CASE_IDS, factors=FACTORS)
    return tasks


def validate_exact_match(tasks, args, *, exact_plan, case_ids, factors):
    import collect_qrte_mean_field_active_dt_scan4 as collector

    plan = qrte.load_json(exact_plan)
    if (plan.get("evolution_mode") != "sparse" or plan["case_ids"] != list(case_ids)
            or plan["factors"] != list(factors)):
        raise ValueError("Expected a matching exact scan plan with the same cases and factors")
    for key in ("n_steps", "overlap_tol", "max_workers"):
        if getattr(args, key) != plan[key]:
            raise ValueError(f"{key} must match the exact scan ({plan[key]})")
    by_index = {r["task_index"]: r for r in plan["tasks"]}
    for task, row in zip(tasks, collector.plan_rows(tasks)):
        if 2 * task["case"]["active_orbitals"] > 12:
            raise ValueError("This scan only permits systems with at most 12 qubits")
        for key in MATCH_FIELDS:
            if row[key] != by_index[task["task_index"]][key]:
                raise ValueError(f"Task {task['task_index']}: {key} differs from exact scan")


def record_comparison(tasks, args, *, exact_plan=None):
    import hashlib

    exact_plan = EXACT_PLAN if exact_plan is None else exact_plan
    qrte.write_json(args.outdir / "exact_comparison.json", {
        "exact_plan_path": str(exact_plan),
        "exact_plan_sha256": hashlib.sha256(exact_plan.read_bytes()).hexdigest(),
        "matched_fields": list(MATCH_FIELDS),
        "matched_stopping_settings": {key: getattr(args, key) for key in ("n_steps", "overlap_tol", "max_workers")},
        "comparison": "Same timestep, cached Hamiltonian, saved HF spectrum, initial determinant and energy reference; propagation changes to order-2 Trotter, one slice",
        "evolution_mode": "trotter", "trotter_order": 2, "trotter_steps": 1,
        "task_count": len(tasks), "new_molecular_calculations": False,
    })


def main():
    base.main(case_ids=CASE_IDS, factors=FACTORS, default_outdir=DEFAULT_OUTDIR,
              description=__doc__, evolution_mode="trotter", task_loader=load_tasks)


if __name__ == "__main__":
    main()
