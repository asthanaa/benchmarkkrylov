# CompressK / `ccik` Documentation

The importable package name is `ccik`. The active repository supports exactly three solvers:

- `ccik`: baseline selected-space CCIK
- `ccik_thick`: selected-space CCIK with thick restart
- `ccik_stochastic`: selected-space CCIK with multidepth stochastic candidate discovery

The runtime representation is sparse over selected determinant pairs by default. Full dense `H|q_k>` formation remains available only through `selector_mode = "full"` for validation and regression checks.

## Start here

- Quickstart: [quickstart.md](quickstart.md)
- Configuration reference: [configuration.md](configuration.md)
- Algorithm details: [algorithms.md](algorithms.md)
- API summary: [api.md](api.md)
- Architecture map: [architecture.md](architecture.md)

## Supporting docs

- Development and testing: [development.md](development.md)
- Legacy scripts: [legacy.md](legacy.md)

## Pointers

- Driver walkthrough: `scripts/README.md`
- Equation-to-code mapping: `src/ccik/README.md`
- Benchmark summary: `experiments/sparse_backend_benchmark_20260312/SUMMARY.md`
