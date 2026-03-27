import pytest


pyscf = pytest.importorskip("pyscf")

import numpy as np

from ccik import (
    build_cas_hamiltonian_pyscf,
    ccik_ground_energy_dense,
    ccik_ground_energy_dense_thick_restart,
    ccik_ground_energy_stochastic,
    exact_cas_fci_energy_pyscf,
    make_mol_pyscf,
)
from ccik.core import generalized_eigh
from ccik.params import CCIKParams, CCIKStochasticParams, CCIKThickRestartParams
from ccik.pyscf_cas import CASSpec


def _build_h2_cas() -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int], float]:
    mol = make_mol_pyscf(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    cas = CASSpec(ncas=2, nelecas=2, ncore=0)
    h1, eri8, ecore, ncas, nelec = build_cas_hamiltonian_pyscf(mol, cas=cas)
    e_fci_tot = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    return h1, eri8, float(ecore), int(ncas), nelec, e_fci_tot


def _build_h4_chain_cas(r_ang: float = 1.5) -> tuple[np.ndarray, np.ndarray, float, int, tuple[int, int], float]:
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
    e_fci_tot = float(ecore + exact_cas_fci_energy_pyscf(h1, eri8, ncas, nelec))
    return h1, eri8, float(ecore), int(ncas), nelec, e_fci_tot


def test_h2_all_three_solvers_close_to_fci() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h2_cas()

    ccik_params = CCIKParams(
        m=8,
        nadd=50,
        nkeep=200,
        Kv=50,
        selector_mode="cipsi",
        selector_depth=1,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        verbose=False,
        orth_tol=1e-12,
    )
    thick_params = CCIKThickRestartParams(
        m_cycle=8,
        ncycles=1,
        nroot=2,
        tol=1e-12,
        nadd=50,
        nkeep=200,
        Kv=50,
        selector_mode="cipsi",
        selector_depth=1,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        verbose=False,
        orth_tol=1e-12,
    )
    stochastic_params = CCIKStochasticParams(
        m=8,
        nadd=50,
        nkeep=200,
        Kv=50,
        selector_mode="stochastic",
        selector_depth=2,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        n_walkers=2000,
        seed=0,
        parent_power=1.0,
        p_double=0.6,
        mixed_double_weight=1.0,
        eps_denom=1e-12,
        verbose=False,
        orth_tol=1e-12,
    )

    st_ccik: dict[str, int] = {}
    st_stoch: dict[str, int] = {}

    e_ccik = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=ccik_params, stats=st_ccik))
    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))
    e_stoch = float(ecore + ccik_ground_energy_stochastic(h1, eri8, ncas, nelec, params=stochastic_params, stats=st_stoch))

    assert abs(e_ccik - e_fci_tot) < 1e-10
    assert abs(e_thick - e_fci_tot) < 1e-10
    assert abs(e_stoch - e_fci_tot) < 1e-10

    assert st_ccik["m_eff"] >= 1
    assert st_ccik["ndet_sum"] >= st_ccik["ndet_union"]
    assert st_stoch["m_eff"] >= 1
    assert st_stoch["ndet_sum"] >= st_stoch["ndet_union"]


def test_stochastic_solver_is_reproducible_with_seed() -> None:
    h1, eri8, ecore, ncas, nelec, _ = _build_h2_cas()
    params = CCIKStochasticParams(
        m=10,
        nadd=10,
        nkeep=50,
        Kv=10,
        selector_mode="stochastic",
        selector_depth=2,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        n_walkers=5000,
        seed=123,
        verbose=False,
        orth_tol=1e-12,
    )
    e1 = float(ecore + ccik_ground_energy_stochastic(h1, eri8, ncas, nelec, params=params))
    e2 = float(ecore + ccik_ground_energy_stochastic(h1, eri8, ncas, nelec, params=params))
    assert np.isclose(e1, e2, atol=0.0, rtol=0.0)


def test_h4_selection_variants_remain_close_to_fci() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h4_chain_cas(r_ang=1.5)

    thick_params = CCIKThickRestartParams(
        m_cycle=12,
        ncycles=1,
        nroot=3,
        tol=1e-10,
        nadd=12,
        nkeep=18,
        Kv=10,
        selector_mode="cipsi",
        selector_depth=1,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        orth_tol=1e-12,
        verbose=False,
    )
    stochastic_params = CCIKStochasticParams(
        m=18,
        nadd=12,
        nkeep=18,
        Kv=10,
        selector_mode="stochastic",
        selector_depth=2,
        selector_ci_coeff_cutoff=1e-8,
        selector_cutoff=1e-8,
        n_walkers=10000,
        seed=0,
        parent_power=1.0,
        p_double=0.6,
        mixed_double_weight=1.0,
        eps_denom=1e-12,
        orth_tol=1e-12,
        verbose=False,
    )

    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))
    e_stoch = float(ecore + ccik_ground_energy_stochastic(h1, eri8, ncas, nelec, params=stochastic_params))

    assert abs(e_thick - e_fci_tot) < 2e-2
    assert abs(e_stoch - e_fci_tot) < 2e-2


def test_full_selector_mode_remains_available() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h2_cas()

    dense_params = CCIKParams(
        m=8,
        nadd=50,
        nkeep=200,
        Kv=50,
        selector_mode="full",
        verbose=False,
        orth_tol=1e-12,
    )
    thick_params = CCIKThickRestartParams(
        m_cycle=8,
        ncycles=1,
        nroot=2,
        tol=1e-12,
        nadd=50,
        nkeep=200,
        Kv=50,
        selector_mode="full",
        verbose=False,
        orth_tol=1e-12,
    )

    e_dense = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=dense_params))
    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))

    assert abs(e_dense - e_fci_tot) < 1e-10
    assert abs(e_thick - e_fci_tot) < 1e-10


def test_threshold_mode_zero_threshold_solver_path_remains_exact_on_h2() -> None:
    h1, eri8, ecore, ncas, nelec, e_fci_tot = _build_h2_cas()

    dense_params = CCIKParams(
        m=8,
        compression_policy="threshold",
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )
    thick_params = CCIKThickRestartParams(
        m_cycle=8,
        ncycles=2,
        nroot=2,
        tol=1e-12,
        compression_policy="threshold",
        store_abs_tol=0.0,
        propagate_abs_tol=0.0,
        candidate_mode="full",
        selector_mode="abs_residual",
        orth_tol=1e-12,
        verbose=False,
    )

    e_dense = float(ecore + ccik_ground_energy_dense(h1, eri8, ncas, nelec, params=dense_params))
    e_thick = float(ecore + ccik_ground_energy_dense_thick_restart(h1, eri8, ncas, nelec, params=thick_params))

    assert abs(e_dense - e_fci_tot) < 1e-10
    assert abs(e_thick - e_fci_tot) < 1e-10


def test_generalized_eigh_filters_near_singular_overlap_modes() -> None:
    H = np.array([[-1.0, -1.0], [-1.0, -1.0]])
    S = np.array([[1.0, 1.0 - 1.0e-14], [1.0 - 1.0e-14, 1.0]])

    evals, X = generalized_eigh(H, S, eps=1.0e-10)

    assert evals.shape == (1,)
    assert X.shape == (2, 1)
    assert abs(float(evals[0]) + 1.0) < 1.0e-10
