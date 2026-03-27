# CompressK / `ccik`

[![Documentation](https://img.shields.io/badge/Documentation-Open%20Docs-0A7EA4?style=for-the-badge)](https://asthanaa.github.io/CompressK/)

This repository contains a Python package in `src/ccik` and runnable drivers in `scripts` for three compressed-Krylov solvers over CAS Hamiltonians:

- `ccik`: baseline selected-space CCIK
- `ccik_thick`: CCIK with thick restart
- `ccik_stochastic`: CCIK with multidepth stochastic candidate discovery

The active implementation stores Krylov vectors sparsely over selected determinant pairs. Full dense `H|q_k>` formation is no longer the default path; it remains available only as the explicit `selector_mode = "full"` validation fallback.

## Documentation

- Hosted docs: https://asthanaa.github.io/CompressK/
- In-repo docs index: `docs/index.md`
- Quickstart: `docs/quickstart.md`
- Configuration reference: `docs/configuration.md`
- Algorithms overview: `docs/algorithms.md`
- Architecture map: `docs/architecture.md`

## Recommended entry points

- Main N2 scan driver: `python scripts/n2_cas_scan.py --config <toml>`
- N2 PES plot driver: `python scripts/n2_pes_scan_and_plot.py --config <toml>`
- Sparse-backend benchmark/tuning driver: `python scripts/benchmark_sparse_backend.py --cases n2_cas10_10`
- Sample config: `tests/n2_sample.toml`

## Current benchmark status

Representative sparse-backend tuning artifacts are in:

- `experiments/sparse_backend_benchmark_20260312/SUMMARY.md`

Main takeaways from that benchmark:

- On N2 / 6-31G / CAS(10,10), tuned `ccik` and tuned `ccik_stochastic` both stayed below `1e-3 Eh` on the tested validation geometries.
- Tuned `ccik_thick` was close but remained slightly above target at the short bond.
- The tuned CAS(10,10) settings do not transfer cleanly to stretched CAS(12,12); larger active spaces still need algorithmic improvement, not just parameter transfer.

## Current scope

The active repository focuses on the three solver variants above. Earlier neural-network selector experiments are not part of the maintained code path.
