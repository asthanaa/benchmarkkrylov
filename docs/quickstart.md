# Quickstart

## Install

Python 3.11 or newer is required.

Minimal install:

```bash
pip install -e .
```

Recommended for the chemistry workflows:

```bash
pip install -e ".[quantum]"
```

Optional test dependencies:

```bash
pip install -e ".[quantum,test]"
```

## Run the sample N2 input

The sample config is `tests/n2_sample.toml`.

Run:

```bash
python scripts/n2_cas_scan.py --config tests/n2_sample.toml
```

The script prints per-geometry energies and writes `out_n2_cas_scan.csv`.

## Choose the solver

The driver accepts one method with `[run].method` or several with `[run].methods`.

Valid method names:

- `ccik`
- `ccik_thick`
- `ccik_stochastic`

Example, single method:

```toml
[run]
method = "ccik_thick"
```

Example, two methods:

```toml
[run]
methods = ["ccik", "ccik_stochastic"]
```

## Benchmark the sparse backend

To run the staged sparse-backend tuning workflow used for the current documentation:

```bash
python scripts/benchmark_sparse_backend.py --cases n2_cas10_10 --methods ccik ccik_thick ccik_stochastic
```

Artifacts are written under `experiments/sparse_backend_benchmark_20260312/`.

## Common gotchas

- For a singlet (`spin = 0`), the drivers check `mol.nelectron == 2*ncore + nelecas`.
- CAS-FCI is used as a reference in the example scripts and becomes expensive quickly as the active space grows.
- `ccik_stochastic` is reproducible only when `seed` is fixed.
- `selector_mode = "full"` is a validation fallback, not the default production path.

## Next

- Configuration details: [configuration.md](configuration.md)
- Algorithm details: [algorithms.md](algorithms.md)
