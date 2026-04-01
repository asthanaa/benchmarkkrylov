"""Beginner example: solve the H2 ground state in the smallest active space.

This script is intentionally short. It shows the full chemistry workflow:

1. build a molecule
2. choose an active space
3. build the CAS Hamiltonian
4. solve it with exact CAS-FCI and with CCIK-thick
5. measure the CCIK error relative to exact CAS-FCI
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


def run_example() -> dict[str, float]:
    """Run a tiny H2 calculation and compare CCIK-thick to exact CAS-FCI."""

    # H2 is the smallest clean example in this repo.
    # The bond length 0.74 Angstrom is close to the equilibrium distance.
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=2, nelecas=2, ncore=0)

    # Build the active-space Hamiltonian H from the molecular integrals.
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

    # On such a tiny problem we can keep very generous fixed budgets.
    params = CCIKThickRestartParams(
        m_cycle=8,
        ncycles=1,
        nroot=2,
        tol=1e-12,
        compression_policy="fixed_budget",
        fixed_store_keep=16,
        fixed_propagate_keep=8,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )

    # Add the frozen-core energy shift ecore back to the active-space answer.
    # We keep the total energies for reference, but the teaching quantity is
    # the error relative to exact CAS-FCI.
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    e_ccik_total = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=params))
    delta_millihartree = 1000.0 * (e_ccik_total - e_fci_total)

    return {
        "e_fci_total": e_fci_total,
        "e_ccik_total": e_ccik_total,
        "delta_millihartree": delta_millihartree,
    }


def main() -> None:
    result = run_example()
    print("H2 example")
    print(f"Exact CAS-FCI total energy : {result['e_fci_total']:+.12f} Eh")
    print(f"CCIK-thick total energy    : {result['e_ccik_total']:+.12f} Eh")
    print(f"Error relative to FCI      : {result['delta_millihartree']:+.6f} mEh")


if __name__ == "__main__":
    main()
