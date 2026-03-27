# Public API

The package exposes a small top-level API from `src/ccik/__init__.py`.

## Solvers

- `ccik_ground_energy_dense(...)`
  Meaning: baseline selected-space CCIK solver.
- `ccik_ground_energy_dense_thick_restart(...)`
  Meaning: selected-space CCIK with thick restart.
- `ccik_ground_energy_stochastic(...)`
  Meaning: selected-space CCIK with multidepth stochastic candidate discovery.

All three return the lowest electronic CAS energy. They do not include `ecore`; the driver scripts add that offset.

## Hamiltonian helpers

- `make_mol_pyscf(...)`
  Meaning: construct a PySCF molecule object.
- `build_cas_hamiltonian_pyscf(...)`
  Meaning: build the CAS one-body integrals, two-body integrals, core offset, and electron counts.
- `exact_cas_fci_energy_pyscf(...)`
  Meaning: compute the exact CAS-FCI reference energy for comparison.

## Parameter dataclasses

- `CCIKParams`
  Used by `ccik_ground_energy_dense`.
- `CCIKThickRestartParams`
  Used by `ccik_ground_energy_dense_thick_restart`.
- `CCIKStochasticParams`
  Used by `ccik_ground_energy_stochastic`.

Each dataclass is the single source of truth for defaults.

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
  Fields:
  - `ncas`: number of active orbitals
  - `nelecas`: number of active electrons
  - `ncore`: number of frozen doubly occupied core orbitals
