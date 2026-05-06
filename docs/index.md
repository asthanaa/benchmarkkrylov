# CompressK Documentation

This documentation is written for two audiences:

- **Chemistry students** who want to understand what CCIK does and run a small example
- **Researchers and developers** who need the benchmarking, Quantum Package, and implementation details

If you are new to the project, follow the beginner path first.

## Start here

1. Read [quickstart.md](quickstart.md)
2. Skim the [glossary.md](glossary.md)
3. Run the `H2` example from [examples.md](examples.md)
4. Use [choose_solver.md](choose_solver.md) to decide which solver to try next

## Beginner path

- [Quickstart](quickstart.md)
- [Chemistry glossary](glossary.md)
- [Choose a solver](choose_solver.md)
- [Examples](examples.md)
- [API](api.md)

## Advanced path

- [Advanced workflows](advanced.md)
- [Reproducible setup](reproducibility.md)
- [Fixed-budget N2 selector results](fixed_budget_results.md)
- [Algorithms](algorithms.md)
- [Quantum Package backend](quantum_package_backend.md)
- [Architecture](architecture.md)
- [Development](development.md)
- [Legacy](legacy.md)

## What this repository contains

- `src/ccik`: the importable Python package
- `examples`: beginner scripts and notebooks
- `docs`: the narrative documentation
- `scripts/advanced`: benchmark, sweep, parity, and QP workflows
- `scripts/_shared`: helpers for the advanced workflows
- `ccik_qp`: vendored Quantum Package plugin source
- `sample_inputs`: tracked teaching inputs and one-command plotting wrappers
- `inputs`: local-only runtime study definitions for QP iteration studies
- `tests`: checks that keep examples, API behavior, and advanced scripts working
