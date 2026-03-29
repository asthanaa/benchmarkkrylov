"""Beginner example: compare the three solver families on an H4 chain."""

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
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_stochastic,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.params import CCIKParams, CCIKStochasticParams, CCIKThickRestartParams  # noqa: E402
from ccik.pyscf_cas import CASSpec  # noqa: E402


def run_example(r_ang: float = 1.5) -> dict[str, float]:
    """Compare dense, thick-restart, and stochastic CCIK on a small H4 chain."""

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
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)

    dense_params = CCIKParams(
        m=12,
        nadd=12,
        nkeep=18,
        Kv=10,
        selector_mode="abs_residual",
        selector_depth=1,
        orth_tol=1e-12,
        verbose=False,
    )
    thick_params = CCIKThickRestartParams(
        m_cycle=12,
        ncycles=1,
        nroot=3,
        tol=1e-10,
        nadd=12,
        nkeep=18,
        Kv=10,
        selector_mode="abs_residual",
        selector_depth=1,
        orth_tol=1e-12,
        verbose=False,
    )
    stochastic_params = CCIKStochasticParams(
        m=18,
        nadd=12,
        nkeep=18,
        Kv=10,
        selector_mode="abs_residual",
        candidate_mode="stochastic",
        selector_depth=2,
        n_walkers=10000,
        seed=0,
        orth_tol=1e-12,
        verbose=False,
    )

    e_fci_total = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    e_dense_total = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=dense_params))
    e_thick_total = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))
    e_stochastic_total = float(ecore + ccik_ground_energy_stochastic(h1, eri8, ncas, nelec, params=stochastic_params))

    return {
        "e_fci_total": e_fci_total,
        "e_dense_total": e_dense_total,
        "e_thick_total": e_thick_total,
        "e_stochastic_total": e_stochastic_total,
        "dense_delta_mEh": 1000.0 * (e_dense_total - e_fci_total),
        "thick_delta_mEh": 1000.0 * (e_thick_total - e_fci_total),
        "stochastic_delta_mEh": 1000.0 * (e_stochastic_total - e_fci_total),
    }


def main() -> None:
    result = run_example()
    print("H4 solver comparison")
    print(f"Exact CAS-FCI total energy : {result['e_fci_total']:+.12f} Eh")
    print(f"CCIK total energy          : {result['e_dense_total']:+.12f} Eh ({result['dense_delta_mEh']:+.4f} mEh)")
    print(f"CCIK-thick total energy    : {result['e_thick_total']:+.12f} Eh ({result['thick_delta_mEh']:+.4f} mEh)")
    print(
        f"CCIK-stochastic total      : {result['e_stochastic_total']:+.12f} Eh "
        f"({result['stochastic_delta_mEh']:+.4f} mEh)"
    )


if __name__ == "__main__":
    main()
