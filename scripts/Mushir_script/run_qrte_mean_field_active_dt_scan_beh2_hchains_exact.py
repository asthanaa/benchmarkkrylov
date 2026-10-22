#!/usr/bin/env python
"""Exact exponential QRTE for BeH2 and H4/H6/H7: 49 conditions, indices 0..48.

Reuses the original seven cases, factors, saved HF widths, cached Hamiltonians,
initial determinants and reference energies. No Trotterization.
"""

import run_qrte_mean_field_active_dt_scan4 as base
from run_qrte_mean_field_active_dt_scan_beh2_hchains import CASE_IDS

qrte, mf = base.qrte, base.mf
FACTORS = base.FACTORS
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_dt_scan_beh2_hchains_exact"
run_task = base.run_task


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="sparse")


def load_tasks(args):
    return base.load_tasks(args, case_ids=CASE_IDS)


def main():
    base.main(case_ids=CASE_IDS, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="sparse")


if __name__ == "__main__":
    main()
