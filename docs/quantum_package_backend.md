# Quantum Package Backend

This page records the current production status of the Quantum Package
integration.

## Repository layout

The maintained QP source path in this repository is:

- `ccik_qp/stable/ccik/`

That directory is the tracked plugin source tree. At runtime, the compiled
plugin binary still lives under your Quantum Package installation:

- `QP_ROOT/plugins/ccik_qp/stable/ccik/ccik`

The shared sync/rebuild helpers are in:

- `scripts/_shared/qp_ccik_common.py`
- `scripts/advanced/qp_ccik_bridge.py`

## Current answer

For the validated large-`N2` runs in this repository, the production path is:

- Quantum Package backend
- `matvec_backend = "connected_sd_fullq"`
- `selector_mode = "abs_residual"`
- `method = "ccik_thick"`
- `compression_policy = "fixed_budget"`

This is the path behind the finished `N2 CAS(12e,12o)` and `CAS(14e,14o)`
benchmark sweeps summarized in [fixed_budget_results.md](fixed_budget_results.md).

## Why this is the production path

The current benchmark evidence is simple:

- the best completed `CAS(12e,12o)` result is a thick QP fixed-budget run with
  `|ΔE| = 0.000783 mEh`
- the best completed `CAS(14e,14o)` result is also a thick QP fixed-budget run,
  with `|ΔE| = 0.710970 mEh`
- both runs use `abs_residual` and the `connected_sd_fullq` backend

So the large-system question has shifted. The main tuning problem is now the
fixed-budget compression pair, not the backend choice and not the selector.

## Recommended QP settings

These are the settings to start from for the current large-`N2` workflow:

```toml
[run]
method = "ccik_thick"

[ccik_thick]
selector_mode = "abs_residual"
compression_policy = "fixed_budget"
fixed_store_keep = 65536
fixed_propagate_keep = 512
```

On the QP side, the backend default should be:

```text
ccik_matvec_backend = connected_sd_fullq
```

The packaged `fixed_propagate_keep = 512` default is a reasonable starting
point, not the final answer for every active space. For the finished large-`N2`
benchmarks, the first tuning move is to scan:

- `fixed_propagate_keep = 512`
- `fixed_propagate_keep = 1024`
- `fixed_propagate_keep = 2048`

while keeping:

- `fixed_store_keep = 65536`

## Compression strategy

The completed `N2` sweeps suggest a stable workflow:

1. keep the QP backend fixed at `connected_sd_fullq`
2. keep `selector_mode = "abs_residual"`
3. keep the solver branch fixed at `ccik_thick`
4. start with `fixed_store_keep = 65536`
5. tune `fixed_propagate_keep` first
6. only increase `fixed_store_keep` if the propagate scan stops improving

That strategy transfers cleanly from the finished `CAS(12e,12o)` benchmark to
the finished `CAS(14e,14o)` benchmark.

## Execution model on this machine

The maintained execution path on this macOS setup is still the direct-binary
route:

- prepare the EZFIO input
- set the CCIK parameters in EZFIO
- run the compiled `ccik` plugin binary directly

This remains more reliable here than relying on `qp run ccik`.

## Representative study inputs

The tracked `inputs/` tree is the source surface for live QP studies.

- small and medium cases may keep several `(Kprop, Kstore)` lines
- the tracked default run pair is:
  - `Kprop=500`, `Kstore=50000`
  - `Kprop=1000`, `Kstore=100000`
- `CAS(14e,16o)` is the exception and keeps:
  - `Kprop=5000`, `Kstore=50000`

The special restart-after-50 case is:

- `inputs/n2/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3/input.json`

Its restart settings are:

- `m_cycle = 51`
- `nroot = 3`

Cycle 1 starts from one seed vector, so the first restart happens after
`51 - 1 = 50` newly generated Krylov vectors.

## What this page is not claiming

This page is intentionally narrow. It is not a claim that:

- every selector ablation has been settled forever
- every molecule should use the same budgets
- every benchmark should be referenced to DMRG

It is only the current production answer supported by the finished benchmark
artifacts in this repository.
