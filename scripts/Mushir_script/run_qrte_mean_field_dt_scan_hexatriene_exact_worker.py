#!/usr/bin/env python
"""Run a worker of the exact hexatriene timestep scan."""

import run_qrte_mean_field_active_dt_scan_upto12_exact_worker as worker
import run_qrte_mean_field_dt_scan_hexatriene_exact as scan


if __name__ == "__main__":
    worker.main(scan_module=scan)
