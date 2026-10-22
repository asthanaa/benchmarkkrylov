#!/usr/bin/env python
"""Ten-system QRTE with mean-field dt=pi/W, first order, and four slices.

Reuses the saved RHF/ROHF fixed-active-Nalpha/Nbeta determinant bandwidth,
validated CASSCF/natural-orbital Hamiltonians, and original QRTE references.
Each full timestep dt is split into four first-order slices of dt/4.
No new SCF, CASSCF, or interacting CAS diagonalization is performed.
"""

from __future__ import annotations

import run_qrte_mean_field_active_pi_over_w_small10 as mean_field

DEFAULT_OUTDIR = mean_field.qrte.ROOT / "runs/qrte_mean_field_active_pi_over_w_small10_trotter1_steps4"


def parse_args():
    args = mean_field.qrte.parse_args(
        default_outdir=DEFAULT_OUTDIR, description=__doc__,
        default_trotter_order=1, default_trotter_steps=4)
    if (args.delta_t is not None or args.evolution_mode != "trotter"
            or args.trotter_order != 1 or args.trotter_steps != 4):
        raise SystemExit("This workflow requires dt=pi/W_MF, --evolution-mode trotter, "
                         "--trotter-order 1 and --trotter-steps 4")
    return args


def main():
    mean_field.run(parse_args())


if __name__ == "__main__":
    main()
