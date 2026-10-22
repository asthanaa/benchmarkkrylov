#!/usr/bin/env python
"""Prepare or collect the <=12-qubit exact scan, reusing validated exact runs."""

import collect_qrte_mean_field_active_dt_scan4 as collector
from collect_qrte_mean_field_active_dt_scan_beh2_hchains import LABELS as CHAIN_LABELS
import run_qrte_mean_field_active_dt_scan_upto12_exact as scan

LABELS = {**collector.LABELS, **CHAIN_LABELS}


def main():
    collector.main(scan_module=scan, labels=LABELS, prepare_tasks=scan.reuse_completed)


if __name__ == "__main__":
    main()
