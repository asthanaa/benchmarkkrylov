#!/usr/bin/env python
"""Matched order-2, one-slice Trotter QRTE scan for hexatriene spins 0 and 2."""

import run_qrte_mean_field_dt_scan_hexatriene_exact as exact
import run_qrte_mean_field_active_dt_scan_upto12_trotter2 as comparison

base, qrte, mf = exact.base, exact.qrte, exact.mf
CASE_IDS, FACTORS = exact.CASE_IDS, exact.FACTORS
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_dt_scan_hexatriene_trotter2"
EXACT_PLAN = exact.DEFAULT_OUTDIR / "scan_plan.json"
run_task = base.run_task


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="trotter")


def load_tasks(args):
    tasks = base.load_tasks(args, case_ids=CASE_IDS, factors=FACTORS)
    if any(t["case"]["active_orbitals"] != 6 or t["case"]["active_electrons"] != 6 for t in tasks):
        raise ValueError("Hexatriene requires the saved 6-electron, 6-orbital (12-qubit) active spaces")
    comparison.validate_exact_match(tasks, args, exact_plan=EXACT_PLAN, case_ids=CASE_IDS, factors=FACTORS)
    return tasks


def record_comparison(tasks, args):
    comparison.record_comparison(tasks, args, exact_plan=EXACT_PLAN)


def main():
    base.main(case_ids=CASE_IDS, factors=FACTORS, default_outdir=DEFAULT_OUTDIR,
              description=__doc__, evolution_mode="trotter", task_loader=load_tasks)


if __name__ == "__main__":
    main()
