"""Beginner-friendly public API for the supported CCIK solvers.

Most new users only need three ideas:

- build a molecule with :func:`make_mol_pyscf`
- build an active-space Hamiltonian with :func:`build_cas_hamiltonian_pyscf`
- solve it with :func:`ccik_ground_energy_dense_thick_restart`

The heavier benchmarking and backend-specific workflows live outside this
package in ``scripts/advanced/``.
"""

from .config import (
    cas_spec_from_dict,
    ccik_params_from_dict,
    ccik_stochastic_params_from_dict,
    ccik_thick_restart_params_from_dict,
    load_config,
    load_default_toml,
    load_toml,
    run_method_from_dict,
    run_methods_from_dict,
)
from .core import ccik_ground_energy_dense
from .params import CCIKParams, CCIKStochasticParams, CCIKThickRestartParams
from .pyscf_cas import build_cas_hamiltonian_pyscf, exact_cas_fci_energy_pyscf, make_mol_pyscf
from .stochastic import ccik_ground_energy_stochastic
from .thick_restart import ccik_ground_energy_dense_thick_restart

__all__ = [
    "CCIKParams",
    "CCIKThickRestartParams",
    "CCIKStochasticParams",
    "ccik_ground_energy_dense",
    "ccik_ground_energy_dense_thick_restart",
    "ccik_ground_energy_stochastic",
    "load_toml",
    "load_default_toml",
    "load_config",
    "ccik_params_from_dict",
    "ccik_thick_restart_params_from_dict",
    "ccik_stochastic_params_from_dict",
    "run_method_from_dict",
    "run_methods_from_dict",
    "cas_spec_from_dict",
    "build_cas_hamiltonian_pyscf",
    "exact_cas_fci_energy_pyscf",
    "make_mol_pyscf",
]
