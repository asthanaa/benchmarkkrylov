# N2 Study Inputs

This folder stores the tracked `N2` study definitions used by:

- `scripts/advanced/n2_thick_iteration_study.py`

The bond-length groups are:

- `1.5/`
  - `R = 1.50 Å`
- `2.0/`
  - `R = 2.00 Å`

Each case folder contains one `input.json` file. That file is the source
definition for the study:

- molecule and basis
- active-space choice
- QP backend settings
- reference policy
- fixed-budget lines such as `Kprop` and `Kstore`

Generated outputs are written back into the same case folder when you run the
study, but those runtime files are local-only and ignored by git.

## Run one case

From the repository root:

```bash
python scripts/advanced/n2_thick_iteration_study.py \
  --input inputs/n2/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3/input.json
```

Another example:

```bash
python scripts/advanced/n2_thick_iteration_study.py \
  --input inputs/n2/1.5/n2_cas10_10_R1p50_m10_c11_r3/input.json
```

The run command also writes the plot files. You do not need a separate plotting
step.

## Exact run-and-open commands

Large-case example (`CAS(14,14)`, two budget lines):

```bash
CASE=inputs/n2/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3
python scripts/advanced/n2_thick_iteration_study.py --input "$CASE/input.json"
open "$CASE/energy_error_vs_iteration_comparison.png"
open "$CASE/k0500_s050000/energy_error_vs_iteration.png"
open "$CASE/k1000_s100000/energy_error_vs_iteration.png"
```

Small/medium example (`CAS(10,10)`, two budget lines):

```bash
CASE=inputs/n2/1.5/n2_cas10_10_R1p50_m10_c11_r3
python scripts/advanced/n2_thick_iteration_study.py --input "$CASE/input.json"
open "$CASE/energy_error_vs_iteration_comparison.png"
open "$CASE/k0500_s050000/energy_error_vs_iteration.png"
open "$CASE/k1000_s100000/energy_error_vs_iteration.png"
```

## Rerun and overwrite local outputs

Use `--force` when you want to rebuild the local runtime outputs for a case:

```bash
python scripts/advanced/n2_thick_iteration_study.py \
  --force \
  --input inputs/n2/2.0/n2_cas14e16_R2p00_m10_c11_r3/input.json
```

Exact rerun-and-open command:

```bash
CASE=inputs/n2/2.0/n2_cas14e16_R2p00_m10_c11_r3
python scripts/advanced/n2_thick_iteration_study.py --force --input "$CASE/input.json"
open "$CASE/energy_error_vs_iteration_comparison.png"
open "$CASE/k5000_s050000/energy_error_vs_iteration.png"
```

## Run every N2 input in this folder tree

The runner discovers `input.json` files recursively, so you can point it at the
whole `inputs/n2/` tree:

```bash
python scripts/advanced/n2_thick_iteration_study.py --inputs-dir inputs/n2
```

If you want to rerun every tracked `N2` case from scratch:

```bash
python scripts/advanced/n2_thick_iteration_study.py --force --inputs-dir inputs/n2
```

## Expected outputs

For each case, the runner writes local runtime files such as:

- `study_summary.json`
- `energy_error_vs_iteration_comparison.png`
- per-run trace folders such as `k0500_s050000/`
- `system_setup/`

The two plot files you will usually open are:

- case-level comparison plot:
  - `CASE/energy_error_vs_iteration_comparison.png`
- one per-run plot:
  - `CASE/k0500_s050000/energy_error_vs_iteration.png`

For the standard two-line cases, the two per-run plot paths are:

- `CASE/k0500_s050000/energy_error_vs_iteration.png`
- `CASE/k1000_s100000/energy_error_vs_iteration.png`

For `CAS(14e,16o)`, the per-run plot path is:

- `CASE/k5000_s050000/energy_error_vs_iteration.png`

Those files are for local monitoring and debugging. The tracked source of truth
in this folder is still just the `input.json` files.

## Large-case convention

For the tracked `N2` studies, the default run pair is:

- `Kprop = 500`
- `Kstore = 50000`
- `Kprop = 1000`
- `Kstore = 100000`

The exception is `CAS(14e,16o)`, which keeps a single fixed-budget line:

- `Kprop = 5000`
- `Kstore = 50000`

The special restart-after-50 case is:

- `inputs/n2/2.0/n2_cas14_14_R2p00_restart_after50_c11_r3/input.json`

That case uses `m_cycle = 51`, so the first thick restart happens after `50`
newly generated Krylov vectors because cycle 1 starts from one seed vector.
