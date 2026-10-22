#!/usr/bin/env python
"""Run one worker of the matched order-2 Trotter timestep scan."""

import run_qrte_mean_field_active_dt_scan_upto12_exact_worker as worker
import run_qrte_mean_field_active_dt_scan_upto12_trotter2 as scan


if __name__ == "__main__":
    worker.main(scan_module=scan)
