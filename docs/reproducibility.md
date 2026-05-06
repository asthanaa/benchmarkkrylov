# Reproducible Setup

This page is the shortest path from a fresh checkout to a working large-space
QP study.

## Fresh checkout: exact commands

From a new machine or a new clone, run these commands in order.

1. Clone the repository and enter it:

```bash
git clone https://github.com/asthanaa/CompressK.git
cd CompressK
```

2. Create and activate the main Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

3. Install the repo package and local plotting/test tools:

```bash
pip install -e ".[quantum,plot,test]"
```

4. Create the dedicated DMRG environment:

```bash
bash scripts/bootstrap_dmrg_env.sh
```

5. Source Quantum Package:

```bash
source /path/to/quantum_package.rc
```

6. Rebuild the vendored plugin inside your Quantum Package installation:

```bash
python scripts/advanced/qp_ccik_bridge.py rebuild-plugin
```

7. Launch the tracked `N2` CAS(10e,10o) and CAS(12e,12o) teaching sample:

```bash
python sample_inputs/n2_error_iterations/run_n2_error_plots.py
```

8. Open the combined CCIK/CKCI-0 plot:

```bash
open sample_inputs/n2_error_iterations/plots/n2_error_vs_iteration_k1000_s050000.png
```

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

The public teaching samples live under `sample_inputs/`. Generated logs, live
plots, runtime folders, and trace files are local-only.

Example:

```bash
python sample_inputs/n2_error_iterations/run_n2_error_plots.py
```

Open the combined CCIK/CKCI-0 plot with:

```bash
open sample_inputs/n2_error_iterations/plots/n2_error_vs_iteration_k1000_s050000.png
```

The tracked sample inputs are:

- `sample_inputs/n2_error_iterations/cases/n2_cas10_10_k1000_s050000/input.json`
- `sample_inputs/n2_error_iterations/cases/n2_cas12_12_k1000_s050000/input.json`

Both use `Kprop = 1000`, `Kstore = 50000`, and CKCI-0 checkpoints every 10
cumulative new Krylov vectors.

## Large-case input policy

For tracked `inputs/` studies, the historical large-case default run pair is:

- `Kprop = 500`
- `Kstore = 50000`
- `Kprop = 1000`
- `Kstore = 100000`

The exception is `CAS(14e,16o)`, which keeps a single fixed-budget line:

- `Kprop = 5000`
- `Kstore = 50000`

This keeps large-case studies cheap enough to reproduce while still matching
the current benchmark/debug workflow.

## Syncing Talon Results

Use the helper from the checkout you want to update:

```bash
bash scripts/advanced/sync_talon_inputs_to_local.sh
```

The default mode copies shareable result files under `inputs/`: JSON summaries,
CSV traces, plots, logs, text notes, TOML files, shell launchers, and captions.
It skips nested `.git/` directories, archives, `.ezfio/work/` runtime tensors,
and anything over 500 MB. To copy every non-excluded file under the 500 MB cap,
set:

```bash
SYNC_ALL_UNDER_LIMIT=1 bash scripts/advanced/sync_talon_inputs_to_local.sh
```
