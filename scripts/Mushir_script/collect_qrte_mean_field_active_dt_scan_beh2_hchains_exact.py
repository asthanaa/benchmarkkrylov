#!/usr/bin/env python
"""Prepare or collect exact exponential BeH2/H-chain QRTE scans; never run QRTE."""

import collect_qrte_mean_field_active_dt_scan4 as collector
from collect_qrte_mean_field_active_dt_scan_beh2_hchains import LABELS
import run_qrte_mean_field_active_dt_scan_beh2_hchains_exact as scan


def main():
    collector.main(scan_module=scan, labels=LABELS)


if __name__ == "__main__":
    main()
