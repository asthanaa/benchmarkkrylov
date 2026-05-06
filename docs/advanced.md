# Advanced Workflows

This page is for research and maintainer workflows, not for the first learning pass.

## Where advanced scripts live

- `scripts/advanced/`: benchmark and study drivers
- `scripts/_shared/`: helpers used by those drivers

Root-level script names still exist as compatibility shims, but the actual implementations now live in the advanced folders above.

## What belongs here

- Quantum Package bridge tooling
- fixed-budget and threshold sweeps
- parity and validation scripts
- large-space `N2` and `Cr2` studies
- overnight campaigns and tuning jobs

## First files to read

- `scripts/README.md`
- `docs/reproducibility.md`
- `docs/fixed_budget_results.md`
- `docs/quantum_package_backend.md`
- `src/ccik/README.md`

## Important distinction

The `examples/` tree is for learning.

The `scripts/advanced/` tree is for:

- benchmarking
- regression studies
- backend validation
- paper-quality figures and workflow automation

## Current benchmark answer

The current settled large-system result in this repository is:

- QP backend
- `connected_sd_fullq`
- `abs_residual`
- `ccik_thick`
- `fixed_budget` compression

The benchmark details and the finished `N2 CAS(12e,12o)` to `CAS(14e,14o)`
transfer story are summarized in [fixed_budget_results.md](fixed_budget_results.md).

## Study inputs

Tracked `sample_inputs/` folders are source definitions, not artifact dumps.

- keep `input.json`
- keep wrapper scripts that produce teaching plots
- keep the case folder names stable
- treat generated plots, traces, logs, and runtime setup folders as local
  output

The local-only `inputs/` workspace can still be used for exploratory QP runs
and larger sweeps.
