# Architecture

## Repository layout

- `src/ccik/`: importable package
- `scripts/`: runnable example drivers and benchmark helpers
- `tests/`: smoke tests and sparse-backend tests
- `configs/`: optional user override configs
- `experiments/`: benchmark artifacts and notes
- `legacy/`: preserved historical scripts

## Core modules

### `ccik.pyscf_cas`

Hamiltonian construction helpers:

- `make_mol_pyscf(...)`
- `build_cas_hamiltonian_pyscf(...)`
- `exact_cas_fci_energy_pyscf(...)`

### `ccik.sparse_ci`

Shared sparse determinant-vector operations:

- sparse Krylov vector storage
- support compression and projection
- sparse inner products and linear combinations
- conversions to and from PySCF selected-CI containers

### `ccik.selection`

Selected-space builders used by all three solvers:

- deterministic selected-CI expansion (`selector_mode = "cipsi"`)
- multidepth stochastic discovery (`selector_mode = "stochastic"`)
- explicit dense fallback (`selector_mode = "full"`)

### `ccik.core`

Baseline selected-space CCIK implementation and shared Ritz helpers:

- `ccik_ground_energy_dense(...)`
- `_score_map(...)`
- `generalized_eigh(...)`

### `ccik.thick_restart`

Thick-restart variant:

- `ccik_ground_energy_dense_thick_restart(...)`
- sparse restart-space construction
- sparse Ritz-vector recycling

### `ccik.stochastic`

Stochastic candidate-discovery variant:

- `ccik_ground_energy_stochastic(...)`

### `ccik.params`

Default parameter dataclasses:

- `CCIKParams`
- `CCIKThickRestartParams`
- `CCIKStochasticParams`

### `ccik.config`

TOML loading and conversion into the parameter dataclasses.

## Drivers

### `scripts/n2_cas_scan.py`

Main example runner:

1. Load config
2. Generate the N2 bond scan
3. Build the CAS Hamiltonian for each geometry
4. Compute CAS-FCI reference
5. Run one or more CCIK solvers
6. Write `out_n2_cas_scan.csv`

### `scripts/n2_pes_scan_and_plot.py`

PES plotting helper for `ccik_thick` and `ccik_stochastic`.

### `scripts/benchmark_sparse_backend.py`

Sparse-backend benchmark and tuning helper.

It stages parameter sweeps on representative geometries, validates tuned settings on a small bond-length set, and writes CSV summaries plus notes into `experiments/`.

### `scripts/cr2_scan.py`

Independent Cr2 CAS-FCI scan helper. It uses the same CAS-building utilities but is not part of the three-solver driver interface.
