"""Beginner example: compare two thick-restart settings on an H4 chain.

This script keeps the teaching story simple:

- exact CAS-FCI is the reference answer
- `ccik_thick` is the default solver to learn first
- a larger thick-restart budget usually lowers the error
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccik import (  # noqa: E402
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense_thick_restart,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402


def run_example(r_ang: float = 1.5) -> dict[str, float]:
    """Compare two thick-restart settings on a small H4 chain."""

    # Place four hydrogen atoms on a straight line with equal spacing.
    # This is still small enough that exact CAS-FCI is practical.
    atom = "; ".join(
        [
            f"H 0 0 {0.0 * r_ang}",
            f"H 0 0 {1.0 * r_ang}",
            f"H 0 0 {2.0 * r_ang}",
            f"H 0 0 {3.0 * r_ang}",
        ]
    )
    mol = make_mol_pyscf(atom=atom, basis="sto-3g", unit="Angstrom", charge=0, spin=0, verbose=0)
    cas = CASSpec(ncas=4, nelecas=4, ncore=0)

    # Build the active-space Hamiltonian that all calculations will use.
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

    # These two settings use explicit fixed budgets.  The first is a smaller
    # budget, and the second is a larger one so students can see how Kstore and
    # Kprop affect the error.
    light_params = CCIKThickRestartParams(
        m_cycle=10,
        ncycles=1,
        nroot=2,
        tol=1e-10,
        compression_policy="fixed_budget",
        fixed_store_keep=28,
        fixed_propagate_keep=14,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )
    refined_params = CCIKThickRestartParams(
        m_cycle=10,
        ncycles=2,
        nroot=2,
        tol=1e-10,
        compression_policy="fixed_budget",
        fixed_store_keep=40,
        fixed_propagate_keep=20,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )

    # Compute one exact answer and two approximate answers from the same
    # active-space Hamiltonian.
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    e_light_total = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=light_params))
    e_refined_total = float(
        ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=refined_params)
    )

    return {
        "e_fci_total": e_fci_total,
        "e_light_total": e_light_total,
        "e_refined_total": e_refined_total,
        "light_delta_mEh": 1000.0 * (e_light_total - e_fci_total),
        "refined_delta_mEh": 1000.0 * (e_refined_total - e_fci_total),
    }


def main() -> None:
    result = run_example()
    print("H4 thick-restart comparison")
    print(f"Exact CAS-FCI total energy : {result['e_fci_total']:+.12f} Eh")
    print(f"Thick, lighter budget      : {result['e_light_total']:+.12f} Eh ({result['light_delta_mEh']:+.4f} mEh)")
    print(
        f"Thick, refined budget      : {result['e_refined_total']:+.12f} Eh "
        f"({result['refined_delta_mEh']:+.4f} mEh)"
    )


if __name__ == "__main__":
    main()
