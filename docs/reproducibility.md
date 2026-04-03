# Reproducible Setup

This page is the shortest path from a fresh checkout to a working large-space
QP study.

## What runs where

- Python backend:
  - small benchmarks
  - teaching examples
  - quick algorithm checks
- QP backend:
  - default path for large active spaces
  - publication-style scans
  - fixed-budget and threshold studies
- DMRG or selected-CI references:
  - large-space reference energies used by the QP study runners

## Repository layout for the QP path

- `ccik_qp/`
  - vendored Quantum Package plugin source tracked in this repository
- `scripts/_shared/qp_ccik_common.py`
  - shared QP install, sync, rebuild, and EZFIO helpers
- `scripts/reference_backends.py`
  - DMRG and selected-CI reference backends
- `scripts/advanced/n2_thick_iteration_study.py`
  - live QP study runner used by the `inputs/` cases

The tracked `ccik_qp/` directory is the canonical plugin source for this
repository snapshot. The compiled plugin binary still lives inside a separate
Quantum Package installation under `QP_ROOT/plugins/ccik_qp/...`.

## Python package install

For the importable Python code and the small examples:

```bash
pip install -e ".[quantum,plot,test]"
```

## DMRG environment bootstrap

Large-space study inputs use a dedicated Python 3.12 environment because the
`block2` wheels used here are not available for the newer repo interpreter.

Create or update that environment with:

```bash
bash scripts/bootstrap_dmrg_env.sh
```

By default this creates:

```text
.conda/cka-dmrg-py312
```

The environment spec is tracked in:

```text
environment.dmrg.yml
```

If you want the environment somewhere else:

```bash
bash scripts/bootstrap_dmrg_env.sh /path/to/env
```

Then point a study at that interpreter with `study.dmrg_python`.

## Quantum Package setup

This repository assumes Quantum Package is installed separately and that
`quantum_package.rc` can be sourced when needed.

Typical workflow:

1. prepare or identify a QP root
2. sync the vendored plugin source into the installed plugin tree
3. rebuild the installed plugin
4. run the study driver

The bridge helper for steps 2 and 3 is:

```bash
python scripts/advanced/qp_ccik_bridge.py rebuild-plugin
```

## Representative QP study

The active `inputs/` tree keeps only study definitions in version control.
Generated logs, live plots, and runtime folders are local-only.

Example:

```bash
python scripts/advanced/n2_thick_iteration_study.py \
  --input inputs/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3/input.json
```

That case demonstrates the restart-after-50 setup:

- `m_cycle = 51`
- `nroot = 3`

Cycle 1 starts from one seed vector, so `m_cycle = 51` means the first restart
happens after `51 - 1 = 50` newly generated Krylov vectors.

## Large-case input policy

For `CAS(14,14)` and larger study inputs, the tracked source definition keeps
only one fixed-budget line unless a new sweep is intentionally being added:

- `Kprop = 500`
- `Kstore = 50000`

This keeps large-case studies cheap enough to reproduce while still matching
the current benchmark/debug workflow.
