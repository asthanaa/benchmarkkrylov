#!/usr/bin/env python
"""Prepare or collect exact exponential four-system QRTE scans; never run QRTE."""

import collect_qrte_mean_field_active_dt_scan4 as collector
import run_qrte_mean_field_active_dt_scan4_exact as scan


def main():
    collector.main(scan_module=scan)


if __name__ == "__main__":
    main()
