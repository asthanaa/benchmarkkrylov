#!/usr/bin/env python
"""Exact exponential QRTE for the original four-system, seven-factor dt scan.

Uses sparse expm_multiply, the original saved HF widths, cached Hamiltonians,
initial determinants and references. Task indices 0..27. No Trotterization.
"""

import run_qrte_mean_field_active_dt_scan4 as base

qrte, mf = base.qrte, base.mf
CASE_IDS, FACTORS = base.CASE_IDS, base.FACTORS
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_dt_scan4_exact"
load_tasks, run_task = base.load_tasks, base.run_task


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="sparse")


def main():
    base.main(case_ids=CASE_IDS, default_outdir=DEFAULT_OUTDIR, description=__doc__, evolution_mode="sparse")


if __name__ == "__main__":
    main()
