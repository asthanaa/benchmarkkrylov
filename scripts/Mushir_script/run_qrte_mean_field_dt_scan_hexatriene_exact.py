#!/usr/bin/env python
"""Exact QRTE timestep scan for hexatriene/cc-pVTZ spins 0 and 2.

Factors 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2 times
pi/W_MF. Twenty-two tasks, indices 0..21; saved HF spectra and CAS caches.
"""

import run_qrte_mean_field_active_dt_scan_upto12_exact as original

base, qrte, mf = original.base, original.qrte, original.mf
FACTORS = original.FACTORS
CASE_IDS = ("hexatriene_equi_cc_pVTZ_0", "hexatriene_equi_cc_pVTZ_2")
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_dt_scan_hexatriene_exact"
run_task = base.run_task


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="sparse")


def load_tasks(args):
    tasks = base.load_tasks(args, case_ids=CASE_IDS, factors=FACTORS)
    if any(t["case"]["active_orbitals"] != 6 or t["case"]["active_electrons"] != 6 for t in tasks):
        raise ValueError("Hexatriene requires the saved 6-electron, 6-orbital (12-qubit) active spaces")
    return tasks


def main():
    base.main(case_ids=CASE_IDS, factors=FACTORS, default_outdir=DEFAULT_OUTDIR,
              description=__doc__, evolution_mode="sparse", task_loader=load_tasks)


if __name__ == "__main__":
    main()
