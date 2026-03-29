# Architecture

## Repository layout

- `src/ccik/`: importable package
- `examples/`: beginner scripts and notebooks
- `docs/`: narrative documentation
- `scripts/advanced/`: research and benchmark drivers
- `scripts/_shared/`: helpers used by advanced drivers
- `tests/`: grouped by teaching examples, solver behavior, and advanced tooling
- `experiments/`: notes and saved artifact locations

## Core package modules

### `ccik.pyscf_cas`

Builds the active-space Hamiltonian from a PySCF molecule.

Start here if you want to understand:

- how the CAS problem is constructed
- what `ecore` means
- how the active-space integrals are produced

### `ccik.core`

Implements the baseline deterministic sparse CCIK iteration.

This is the clearest place to learn the basic Krylov loop.

### `ccik.thick_restart`

Implements the recommended thick-restart variant.

Read this after `ccik.core` if you want to understand why the repository recommends `ccik_thick`.

### `ccik.stochastic`

Implements stochastic candidate discovery.

This is more advanced than the baseline and thick-restart deterministic flows.

### `ccik.selection`

Builds the determinant space used to approximate `H q_k`.

This is one of the most technical modules in the repo and is usually not the first place a new reader should start.

### `ccik.sparse_ci`

Defines the sparse determinant-vector data structures and helper operations.

### `ccik.config`

Parses TOML-style configuration into the parameter dataclasses.

### `ccik.params`

Stores the solver defaults in dataclass form.

## Suggested reading order

1. `ccik.pyscf_cas`
2. `ccik.core`
3. `ccik.thick_restart`
4. `ccik.params`
5. `ccik.selection` and `ccik.sparse_ci`
