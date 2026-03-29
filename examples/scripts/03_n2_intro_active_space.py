"""Beginner example: introduce an active-space calculation on N2.

For this tiny ``CAS(6e,6o)`` teaching problem, the dense CCIK solver reaches
sub-millihartree agreement with exact CAS-FCI on a laptop. That makes it a
better notebook choice than a more aggressively compressed thick-restart setup,
which is the solver to switch back to for larger active spaces.
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
    ccik_ground_energy_dense,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402


def make_n2_atom(distance_ang: float) -> str:
    """Return a PySCF geometry string for N2 at the requested bond length."""

    z = float(distance_ang) / 2.0
    return f"""
    N 0.0 0.0 {-z}
    N 0.0 0.0 {+z}
    """


def run_example(distance_ang: float = 1.10) -> dict[str, float]:
    """Run a small N2 active-space example that reaches the 1 mHa target.

    This example uses dense CCIK because the active space is still small enough
    that the dense solver reproduces exact CAS-FCI to well below ``1 mEh``.
    """

    mol = make_mol_pyscf(
        atom=make_n2_atom(distance_ang),
        basis="6-31g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=6, nelecas=6, ncore=2)
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

    # For this small active space the baseline dense solver is simple and
    # already chemically accurate, so it is the clearest notebook default.
    params = CCIKParams(verbose=False)

    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    e_dense_total = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=params))
    delta_millihartree = 1000.0 * (e_dense_total - e_fci_total)

    return {
        "distance_ang": distance_ang,
        "ncas": float(cas.ncas),
        "nelecas": float(cas.nelecas),
        "solver_name": "ccik_dense",
        "solver_label": "CCIK dense",
        "target_millihartree": 1.0,
        "meets_target": abs(delta_millihartree) < 1.0,
        "e_fci_total": e_fci_total,
        "e_dense_total": e_dense_total,
        "e_ccik_total": e_dense_total,
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
    print(f"Difference                 : {result['delta_millihartree']:+.6f} mEh")
    print(f"Chemical-accuracy target   : < {result['target_millihartree']:.1f} mEh ({result['meets_target']})")


if __name__ == "__main__":
    main()
