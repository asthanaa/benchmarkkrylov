# Public API

The top-level API is intentionally small.

## Solvers

### `ccik_ground_energy_dense(...)`

Plain-language purpose:
- run the baseline deterministic CCIK solver

Chemistry meaning:
- builds a compressed Krylov space over a CAS Hamiltonian without thick restart

### `ccik_ground_energy_dense_thick_restart(...)`

Plain-language purpose:
- run the recommended thick-restart CCIK solver

Chemistry meaning:
- reuses several good Ritz vectors during restart, which is why it is the best default learning path in this repository

### `ccik_ground_energy_stochastic(...)`

Plain-language purpose:
- run the stochastic candidate-discovery variant

Chemistry meaning:
- explores important determinants using stochastic proposals instead of only deterministic discovery

All three return the **electronic CAS energy** only. Add `ecore` if you want the total energy.

## Hamiltonian builders

### `make_mol_pyscf(...)`

- build a PySCF molecule object

### `build_cas_hamiltonian_pyscf(...)`

- build the effective one-electron and two-electron CAS Hamiltonian pieces
- return `h1`, `eri8`, `ecore`, `ncas`, and the active-space electron tuple

### `exact_cas_fci_energy_pyscf(...)`

- compute the exact CAS-FCI electronic energy used as a small-system reference

## Parameter dataclasses

- `CCIKParams`
- `CCIKThickRestartParams`
- `CCIKStochasticParams`

These dataclasses define the solver defaults. For most new users, `CCIKThickRestartParams` is the one to read first.

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

These are mainly useful for config-driven workflows and advanced scripts.
