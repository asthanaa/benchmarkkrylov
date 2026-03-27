# Architecture

## Repository layout

- `src/ccik/`: importable package
- `scripts/`: runnable drivers and benchmark helpers
- `tests/`: smoke tests and sparse-backend tests
- `configs/`: example user configs
- `experiments/`: generated benchmark artifacts and notes
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
- conversion to and from PySCF selected-CI objects

### `ccik.selection`

Selected-space builders used by all three solvers:

- deterministic selected-CI expansion (`selector_mode = "cipsi"`)
- multidepth stochastic discovery (`selector_mode = "stochastic"`)
- explicit dense fallback (`selector_mode = "full"`)

### `ccik.core`

Baseline selected-space CCIK:

- `ccik_ground_energy_dense(...)`
- generalized Ritz helpers
- sparse Krylov update logic

### `ccik.thick_restart`

Thick-restart variant on the same sparse backend:

- `ccik_ground_energy_dense_thick_restart(...)`
- sparse restart-space construction
- thick-restart Ritz recycling

### `ccik.stochastic`

Stochastic candidate-discovery variant:

- `ccik_ground_energy_stochastic(...)`

### `ccik.params`

Parameter dataclasses:

- `CCIKParams`
- `CCIKThickRestartParams`
- `CCIKStochasticParams`

### `ccik.config`

TOML loading and conversion into those parameter dataclasses.

## Drivers

### `scripts/n2_cas_scan.py`

Main N2 scan runner:

1. Load config
2. Generate the bond scan
3. Build the CAS Hamiltonian
4. Compute CAS-FCI reference
5. Run one or more CCIK solvers
6. Write `out_n2_cas_scan.csv`

### `scripts/n2_pes_scan_and_plot.py`

PES plotting helper for `ccik_thick` and `ccik_stochastic`.

### `scripts/benchmark_sparse_backend.py`

Benchmark and tuning helper for the sparse backend.

It stages parameter sweeps on representative geometries, validates tuned settings on a small bond-length set, and writes CSV summaries plus notes into `experiments/`.

### `scripts/cr2_scan.py`

Independent Cr2 CAS-FCI scan using the same CAS-building utilities.
