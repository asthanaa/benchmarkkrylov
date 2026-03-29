# `src/ccik/`

This README is part of the **advanced implementation track**.

If you are new to the project, start with:

- `README.md`
- `docs/quickstart.md`
- `examples/`

## What this package contains

- `pyscf_cas.py`: build active-space Hamiltonians from PySCF
- `core.py`: baseline deterministic CCIK
- `thick_restart.py`: recommended thick-restart variant
- `stochastic.py`: stochastic candidate-discovery variant
- `selection.py`: selected-space construction logic
- `sparse_ci.py`: sparse determinant-vector data structures
- `params.py`: solver default dataclasses
- `config.py`: TOML/config helpers

## Suggested implementation reading order

1. `pyscf_cas.py`
2. `core.py`
3. `thick_restart.py`
4. `params.py`
5. `selection.py`
6. `sparse_ci.py`

## Relationship to the examples

The files in `examples/` show how to call the public API without needing to understand every internal detail here first.
