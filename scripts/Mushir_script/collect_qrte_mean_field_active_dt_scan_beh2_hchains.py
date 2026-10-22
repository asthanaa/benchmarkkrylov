#!/usr/bin/env python
"""Prepare or collect the BeH2/H4/H6/H7 QRTE timestep scan; never run QRTE."""

import collect_qrte_mean_field_active_dt_scan4 as collector
import run_qrte_mean_field_active_dt_scan_beh2_hchains as scan

LABELS = {
    "BeH2_equi_cc_pVTZ_0": "BeH₂/cc-pVTZ, spin 0",
    "BeH2_equi_cc_pVTZ_2": "BeH₂/cc-pVTZ, spin 2",
    "H4_near_equi_sto3g_0": "H4/STO-3G, spin 0",
    "H6_near_equi_sto3g_0": "H6/STO-3G, spin 0",
    "H6_near_equi_cc_pVTZ_0": "H6/cc-pVTZ, spin 0",
    "H6_near_equi_cc_pVTZ_2": "H6/cc-pVTZ, spin 2",
    "H7_near_equi_sto3g_1": "H7/STO-3G, spin 1",
}


def main():
    collector.main(scan_module=scan, labels=LABELS)


if __name__ == "__main__":
    main()
