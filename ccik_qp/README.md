# `ccik_qp`

Vendored Quantum Package plugin source for the `ccik` algorithm.

This directory is now tracked directly inside the main `CompressK` repository so
the Python drivers, the benchmark harness, and the QP plugin source move
together in one reproducible snapshot.

## Provenance

The plugin source originally lived in a separate upstream repository:

- `https://github.com/asthanaa/ccik_qp`

For this repository snapshot, the tracked copy under:

- `CompressK/ccik_qp`

is the canonical source used by the maintained `CompressK` workflows.

## Scope

This vendored source tree focuses on:

- baseline `ccik` only
- no thick restart
- no stochastic selector
- determinant-driven projected `H q_k` construction
- Quantum Package-native execution via `qp run ccik file.ezfio`

## Structure

- `stable/ccik/`
  - Quantum Package plugin module
  - executable: `qp run ccik`
  - plugin parameters in `EZFIO.cfg`

## Current status

The files in `stable/ccik/` are a scaffold for the plugin entry point and its
EZFIO interface. The determinant-driven kernel is the next implementation step.

The plan is:

1. Use Quantum Package determinant infrastructure for generator/selector splits.
2. Build projected `H q_k` determinant-by-determinant instead of using PySCF
   selected-CI contractions.
3. Cache projected `H q_j` data for the Krylov Ritz step.
4. Keep `CompressK` as the front-end for scans, plotting, and validation.

## Intended installation

Once Quantum Package is available:

1. Register the plugin repository with `qp_plugins`
2. Install the `ccik` plugin
3. Run on an EZFIO dataset

Typical workflow:

```bash
qp_plugins download /path/to/ccik_qp
qp_plugins install ccik
qp set_file molecule.ezfio
qp set ccik ccik_m 32
qp run ccik molecule.ezfio
```

The bridge script in `CompressK/scripts/qp_ccik_bridge.py` automates those
steps from the Python side.
