#!/usr/bin/env python
"""Seven-factor mean-field timestep scan for BeH2 and H4/H6/H7 chains.

BeH2/cc-pVTZ spins 0/2; H4/STO-3G spin 0; H6/STO-3G spin 0;
H6/cc-pVTZ spins 0/2; H7/STO-3G spin 1. Factors: 0.25, 0.5, 0.75,
1, 1.25, 1.5, 2. Second-order Trotter, one slice. Task indices 0..48.
Only saved HF spectra and validated cached Hamiltonians are used.
"""

from __future__ import annotations

import run_qrte_mean_field_active_dt_scan4 as base

qrte = base.qrte
mf = base.mf
FACTORS = base.FACTORS
SPECTRUM_SCOPE = base.SPECTRUM_SCOPE
DEFAULT_OUTDIR = qrte.ROOT / "runs/qrte_mean_field_active_dt_scan_beh2_hchains_trotter2"
CASE_IDS = (
    "BeH2_equi_cc_pVTZ_0",
    "BeH2_equi_cc_pVTZ_2",
    "H4_near_equi_sto3g_0",
    "H6_near_equi_sto3g_0",
    "H6_near_equi_cc_pVTZ_0",
    "H6_near_equi_cc_pVTZ_2",
    "H7_near_equi_sto3g_1",
)


def parse_args(argv=None):
    return base.parse_args(argv, default_outdir=DEFAULT_OUTDIR, description=__doc__)


def load_tasks(args):
    return base.load_tasks(args, case_ids=CASE_IDS)


def main():
    base.main(case_ids=CASE_IDS, default_outdir=DEFAULT_OUTDIR, description=__doc__)


if __name__ == "__main__":
    main()
