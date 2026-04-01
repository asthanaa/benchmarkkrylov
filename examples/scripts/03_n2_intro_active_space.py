"""Beginner example: introduce an active-space calculation on N2.

This example keeps the notebook teaching rule simple: use `ccik_thick` by
default and report its error relative to exact CAS-FCI.
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


def make_n2_atom(distance_ang: float) -> str:
    """Return a PySCF geometry string for N2 at the requested bond length."""

    z = float(distance_ang) / 2.0
    return f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """


def run_example(distance_ang: float = 1.10) -> dict[str, float]:
    """Run a small N2 active-space example with thick-restart CCIK."""

    # N2 is a realistic molecule, but CAS(6e,6o) is still small enough that we
    # can explain every step without turning the example into a benchmark.
    mol = make_mol_pyscf(
        atom=make_n2_atom(distance_ang),
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=6, nelecas=6, ncore=2)

    # The active-space Hamiltonian is the chemistry object the CCIK solvers use.
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

    # This thick-restart setup uses explicit fixed budgets.  The values below
    # were tuned for this small teaching case so the error stays well below the
    # 1 mEh chemistry target.
    params = CCIKThickRestartParams(
        m_cycle=20,
        ncycles=6,
        nroot=4,
        tol=1e-10,
        compression_policy="fixed_budget",
        fixed_store_keep=1024,
        fixed_propagate_keep=512,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )

    # Add the frozen-core energy back so we compare total energies, then report
    # the error relative to exact CAS-FCI.
    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    e_thick_total = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=params))
    delta_millihartree = 1000.0 * (e_thick_total - e_fci_total)

    return {
        "distance_ang": distance_ang,
        "ncas": float(cas.ncas),
        "nelecas": float(cas.nelecas),
        "solver_name": "ccik_thick",
        "solver_label": "CCIK thick restart",
        "target_millihartree": 1.0,
        "meets_target": abs(delta_millihartree) < 1.0,
        "e_fci_total": e_fci_total,
        "e_thick_total": e_thick_total,
        "e_ccik_total": e_thick_total,
        "delta_millihartree": delta_millihartree,
    }


def main() -> None:
    result = run_example()
    print("N2 active-space introduction")
    print(f"Bond length                : {result['distance_ang']:.2f} Angstrom")
    print("Chosen active space        : CAS(6e,6o) with 2 frozen core orbitals")
    print(f"Solver used                : {result['solver_label']}")
    print(f"Exact CAS-FCI total energy : {result['e_fci_total']:+.12f} Eh")
    print(f"{result['solver_label']:<26}: {result['e_ccik_total']:+.12f} Eh")
    print(f"Error relative to FCI      : {result['delta_millihartree']:+.6f} mEh")
    print(f"Chemical-accuracy target   : < {result['target_millihartree']:.1f} mEh ({result['meets_target']})")


if __name__ == "__main__":
    main()
