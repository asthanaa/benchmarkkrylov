#!/usr/bin/env python
"""Overlay completed exact and second-order Trotter hexatriene timestep scans."""

import plot_qrte_mean_field_dt_scan_upto12_exact_vs_trotter2 as overlay
import run_qrte_mean_field_dt_scan_hexatriene_exact as exact
import run_qrte_mean_field_dt_scan_hexatriene_trotter2 as trotter
from collect_qrte_mean_field_dt_scan_hexatriene_exact import LABELS


def main():
    overlay.main(exact_module=exact, trotter_module=trotter, labels=LABELS)


if __name__ == "__main__":
    main()
