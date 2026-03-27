# `scripts/`

This directory contains runnable drivers built on top of the package in
`src/ccik` and the external Quantum Package plugin.

## Python prerequisites

- Python `>= 3.11`
- PySCF

From the repo root:

```bash
pip install -e ".[quantum]"
```

Optional test extras:

```bash
pip install -e ".[quantum,test]"
```

## Main Python workflows

### Exact/reference N2 scans

Use the exact-reference scan drivers when validating dense or thick CCIK
against CAS-FCI.

Examples:

```bash
python scripts/n2_accuracy_gate.py
python scripts/n2_pes_scan_and_plot.py --config configs/config.toml --npoints 20
```

### Selector/matvec validation

The maintained selector/parity workflow is:

```bash
python scripts/debug_selector_parity.py --case n2_cas10_10_R1p10 --with-qp
```

What it does:

- syncs the workspace QP plugin source into the installed `QP_ROOT` plugin tree
- rebuilds the plugin with `/bin/bash`
- rejects the run if the rebuilt binary does not print the expected threshold
  parameters
- rejects the run if the trace does not expose the compression-policy field
- compares Python and QP traces on the same active-space Hamiltonian

Important flags:

- `--case`
- `--sweep donor_n2`
- `--selector-mode`
- `--matvec-backend`
- `--compression-policy`
- `--fixed-store-keep`
- `--fixed-propagate-keep`
- `--store-abs-tol`
- `--propagate-abs-tol`
- `--with-qp`
- `--qp-root`

### Fixed-budget workflow

Fixed-budget mode is the recommended large-space sparse policy and uses:

- `compression_policy = "fixed_budget"`
- `fixed_store_keep`
- `fixed_propagate_keep`

The parity/debug tooling now accepts these knobs directly:

- `scripts/debug_selector_parity.py`
  - `--compression-policy fixed_budget`
  - `--fixed-store-keep 1000000`
  - `--fixed-propagate-keep 10000`

The existing large-space threshold studies remain on disk as historical
artifacts, but future large-space donor-backed benchmarking should prefer the
fixed-budget policy.

For dedicated large-space fixed-budget sweeps, use:

- `scripts/n2_qp_fixed_budget_sweep_plots.py`
  - sweeps `fixed_store_keep` while holding `fixed_propagate_keep`
  - sweeps `fixed_propagate_keep` while holding `fixed_store_keep`
  - defaults to `fixed_store_keep = 1000000`
  - defaults to `fixed_propagate_keep = 10000`
  - reuses the same QP rebuild/root validation/reference/calibration path as the large-space threshold driver

### Threshold-mode benchmark workflow

Threshold mode is explicit-only and uses:

- `compression_policy = "threshold"`
- `store_abs_tol`
- `propagate_abs_tol`

The validation/plot workflow is:

1. run threshold validation and QP parity gates
2. only after those pass, run the threshold sweep study

The threshold sweep study for `N2 / 6-31G / CAS(10e,8o)` writes:

- CSV/JSON data tables
- `store_abs_tol` sweep figures
- `propagate_abs_tol` sweep figures

Both validation and sweep drivers are resume-safe:

- `scripts/threshold_mode_validation.py`
  - reuses completed per-case JSON summaries and gate files
  - supports `--force` to invalidate cache
- `scripts/n2_threshold_sweep_plots.py`
  - reuses existing CSV rows and only computes missing sweep points
  - supports `--force` to invalidate cache

For the large-space donor-backed threshold studies, use the dedicated QP
driver instead of the Python-only sweep path:

- `scripts/n2_qp_threshold_sweep_plots.py`
  - `--case n2_cas12_12_R2p50`
  - `--case n2_cas14_14_R2p50`
  - rebuilds the installed plugin first
  - runs a preflight parity gate first
  - calibrates dense/thick structural settings at zero thresholds
  - caches per-threshold run directories, CSV rows, and plot outputs

Canonical QP roots for rebuildable workflows must not live under `/private/tmp`.
Use a stable install root with:

- `quantum_package.rc`
- `config/build.ninja`
- `plugins/ccik_qp/stable/ccik`

## Quantum Package helpers

### Shared helper module

`scripts/qp_ccik_common.py` contains the canonical QP helper path:

- discover `QP_ROOT`
- prepare donor/backed EZFIOs
- write current CCIK parameters
- sync the workspace plugin source into the installed QP tree
- rebuild the installed plugin with `/bin/bash`

### Bridge CLI

Use the bridge to inspect or manipulate the external plugin:

```bash
python scripts/qp_ccik_bridge.py status
python scripts/qp_ccik_bridge.py sync-plugin
python scripts/qp_ccik_bridge.py rebuild-plugin
python scripts/qp_ccik_bridge.py set-params /path/to/file.ezfio --ccik-compression-policy threshold --ccik-store-abs-tol 1e-5 --ccik-propagate-abs-tol 1e-6
python scripts/qp_ccik_bridge.py run /path/to/file.ezfio --mode direct
```

Notes:

- `--mode direct` is the recommended execution path on the current macOS setup
- `rebuild-plugin` uses the installed plugin tree inside `QP_ROOT`, not the
  workspace tree directly
- the threshold parameters are now first-class bridge arguments

## Current production defaults

For new QP runs, the maintained backend is:

- `ccik_matvec_backend = connected_sd_fullq`

The maintained selector default is:

- `ccik_selector_mode = abs_residual`

The maintained compression default is still:

- `ccik_compression_policy = topk`

Threshold mode is available but not the default.

## Canonical current artifacts

Threshold-mode experiment artifacts now live under `experiments/threshold_test`.
Use these directories as the maintained current references:

- `experiments/threshold_test/donor_n2_parity_sweep_20260320_rerun`
- `experiments/threshold_test/threshold_mode_validation_20260325`
- `experiments/threshold_test/n2_cas10e8_threshold_sweep_20260324_v2`
- `experiments/threshold_test/n2_cas12_method_compare_20260324`
- `experiments/threshold_test/n2_qp_n2_cas12_12_R2p50_threshold_sweep_20260325_pristine`
- `experiments/threshold_test/n2_qp_n2_cas14_14_R2p50_threshold_sweep_20260325_pristine`

Future large-space donor-backed sweeps should prefer fixed-budget artifacts
produced by `scripts/n2_qp_fixed_budget_sweep_plots.py`.

Canonical QP roots for threshold work should live outside `/private/tmp`.
