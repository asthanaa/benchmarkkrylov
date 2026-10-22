#!/usr/bin/env python
"""Prepare or collect the matched second-order Trotter hexatriene scan."""

import collect_qrte_mean_field_active_dt_scan4 as collector
from collect_qrte_mean_field_dt_scan_hexatriene_exact import LABELS
import run_qrte_mean_field_dt_scan_hexatriene_trotter2 as scan


def main():
    collector.main(scan_module=scan, labels=LABELS, prepare_tasks=scan.record_comparison)


if __name__ == "__main__":
    main()
