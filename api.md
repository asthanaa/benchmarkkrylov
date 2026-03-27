# API

The package exposes a small top-level API from `src/ccik/__init__.py`.

## Solvers

- `ccik_ground_energy_dense(...)`
  - baseline selected-space CCIK solver
- `ccik_ground_energy_dense_thick_restart(...)`
  - selected-space CCIK with thick restart
- `ccik_ground_energy_stochastic(...)`
  - selected-space CCIK with multidepth stochastic candidate discovery

All three return the lowest electronic CAS energy. They do not include `ecore`.

## Hamiltonian helpers

- `make_mol_pyscf(...)`
- `build_cas_hamiltonian_pyscf(...)`
- `exact_cas_fci_energy_pyscf(...)`

## Parameter dataclasses

- `CCIKParams`
- `CCIKThickRestartParams`
- `CCIKStochasticParams`

These dataclasses define the default values for solver settings.

## Config helpers

- `load_toml(path)`
- `load_default_toml()`
- `load_config(path_or_none)`
- `run_method_from_dict(run_dict)`
- `run_methods_from_dict(run_dict)`
- `ccik_params_from_dict(d)`
- `ccik_thick_restart_params_from_dict(d)`
- `ccik_stochastic_params_from_dict(d)`
- `cas_spec_from_dict(d)`

## CAS specification

- `CASSpec`
  - `ncas`
  - `nelecas`
  - `ncore`
