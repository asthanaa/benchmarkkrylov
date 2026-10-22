#!/usr/bin/env python
"""Prepare or collect the exact hexatriene scan without running QRTE."""

import collect_qrte_mean_field_active_dt_scan4 as collector
import run_qrte_mean_field_dt_scan_hexatriene_exact as scan

LABELS = {scan.CASE_IDS[0]: "Hexatriene/cc-pVTZ, spin 0", scan.CASE_IDS[1]: "Hexatriene/cc-pVTZ, spin 2"}


def main():
    collector.main(scan_module=scan, labels=LABELS)


if __name__ == "__main__":
    main()
