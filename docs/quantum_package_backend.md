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

## Notebook-style dataflow walkthrough

The easiest way to understand the QP backend is to treat it like a short
teaching notebook with two layers:

- outer layer: Python prepares a QP case and launches the plugin
- inner layer: the Fortran plugin runs thick-restart CCIK inside QP

### Cell 1: Python defines the chemistry and solver knobs

In a live study input such as:

- `inputs/n2/2.0/n2_cas12_12_R2p00_m10_c11_r3/input_qp.py`

the editable block does two jobs:

- define the physical system: geometry, basis, active-space sizes
- define the CCIK run shape: `m_cycle`, `n_cycles`, `nroot`, and fixed-budget caps

At this stage nothing numerical has happened inside Quantum Package yet. Python
is only building a specification for the upcoming run.

### Cell 2: Python locates the local Quantum Package installation

The bridge helpers in:

- `scripts/_shared/qp_ccik_common.py`

resolve the local QP installation from `QP_ROOT` and identify the tools needed
for the run:

- `quantum_package.rc`
- `qp_create_ezfio`
- `scf`
- the compiled `ccik` plugin binary

This is the point where the repository stops being "pure Python" and starts
using Quantum Package as the runtime host.

### Cell 3: Python builds an EZFIO case

`run_qp_setup(...)` creates a fresh EZFIO directory and populates it with the
molecule and SCF data. The flow is:

1. run `qp_create_ezfio` from the XYZ geometry
2. fill in any missing EZFIO defaults
3. run `qp set_file <case.ezfio>`
4. request AO and MO two-electron integrals to be written
5. run the QP `scf` binary
6. mark each MO as `Core`, `Active`, or `Deleted`

After this step, the EZFIO tree contains the QP-side case database:

- geometry
- orbitals
- electron counts
- SCF wavefunction
- integral control files
- active-space metadata

### Cell 4: Python writes the CCIK plugin parameters into EZFIO

The plugin parameters are declared in:

- `ccik_qp/stable/ccik/EZFIO.cfg`

Examples include:

- `ccik_fixed_store_keep`
- `ccik_fixed_propagate_keep`
- `ccik_restart_m_cycle`
- `ccik_restart_nroot`
- `ccik_print_iteration_error`

Each of those entries uses:

- `interface: ezfio,provider,ocaml`

which means QP turns the EZFIO fields into live values that the plugin can read
directly at runtime.

The bridge function `set_ccik_params(...)` simply writes the chosen values into:

- `case.ezfio/ccik/...`

So the Python layer is not "passing arguments" to the solver in the usual
command-line sense. It is writing the plugin configuration into the EZFIO case
that QP already knows how to expose.

### Cell 5: Python launches the compiled plugin binary

The maintained execution path is:

```text
source quantum_package.rc
qp set_file case.ezfio
ccik case.ezfio
```

The key command is:

- `qp set_file case.ezfio`

This tells Quantum Package which EZFIO case is the current working case for the
plugin run.

### Cell 6: The plugin entry point decides which solver branch to run

The plugin entry point is:

- `ccik_qp/stable/ccik/ccik.irp.f`

It prints the banner and parameters, then chooses:

- `ccik_thick_restart_driver` when restart mode is enabled
- `ccik_baseline_driver` otherwise

So the user-facing Python script is only the setup/launch layer. The real QP
solve begins here.

### Cell 7: The Fortran driver asks QP for the current case data

Inside `ccik_thick_restart_driver`, the important initialization lines are:

- `PROVIDE all_mo_integrals psi_det psi_coef`
- `call ccik_capture_current_vector(q0)`

This is the core connection between CCIK and Quantum Package:

- `all_mo_integrals` comes from QP's molecular-integral machinery
- `psi_det` and `psi_coef` are QP's working CI wavefunction arrays

`ccik_capture_current_vector(q0)` copies the current QP wavefunction into
CCIK's own sparse-vector format.

At that moment:

- QP owns the case and providers
- CCIK owns the algorithm and sparse-vector manipulations

### Cell 8: One thick-restart cycle inside QP

The inner dataflow for one cycle is:

```text
restart_basis
  -> copy into qbasis
  -> for each basis vector:
       load vector into QP working wavefunction (psi_det, psi_coef)
       build projected H q
  -> solve the Ritz problem in span(qbasis)
  -> print current energy
  -> generate next Krylov vector
  -> load it into QP working wavefunction
  -> build projected H q for the new vector
  -> solve Ritz problem again
  -> repeat until n_basis = m_cycle
  -> keep the best nroot Ritz vectors
  -> store them as restart_basis for the next cycle
```

The most important "bridge back into QP" step is:

- `ccik_load_vector_into_working_wf(vec)`

That subroutine overwrites QP's current `N_det`, `psi_det`, and `psi_coef`
arrays with the vector CCIK wants QP to act on next. In other words:

- CCIK proposes a sparse vector
- QP treats it as the current working wavefunction
- QP-backed routines build the projected Hamiltonian action for that vector

So the plugin repeatedly moves data in both directions:

- QP wavefunction arrays -> CCIK sparse vector
- CCIK sparse vector -> QP wavefunction arrays

### Cell 9: Restart and convergence

At the end of the cycle, the driver:

1. solves the current small Ritz problem
2. compares the cycle energy with the previous cycle energy
3. if not converged, builds the lowest `nroot` Ritz vectors
4. uses those vectors as the restart seeds for the next cycle

That is why thick restart keeps the important low-energy directions without
keeping the full old Krylov basis in memory.

### Cell 10: Final results go back through QP

At the end of the run, the plugin prints:

- `CCIK_FINAL_ELECTRONIC_ENERGY`
- `CCIK_FINAL_TOTAL_ENERGY`
- `CCIK_FINAL_BASIS`

If `ccik_save_final_wf = True`, the final Ritz vector is also written back into
the EZFIO wavefunction. That lets later QP tools reuse the result, for example:

- PT2-like post-corrections
- orbital optimization
- follow-on workflows that want the converged sparse CI state

## End-to-end dataflow summary

```text
Python study input
  -> build XYZ text
  -> compute optional external reference energy (PySCF, for plotting/reporting)
  -> create EZFIO case through QP
  -> run SCF through QP
  -> label active space in EZFIO
  -> write CCIK plugin parameters into EZFIO
  -> source quantum_package.rc
  -> qp set_file case.ezfio
  -> launch compiled ccik plugin

QP plugin runtime
  -> read plugin parameters from EZFIO-backed providers
  -> read integrals and current wavefunction from QP providers
  -> copy QP wavefunction into CCIK sparse-vector format
  -> run thick-restart Krylov / Ritz iterations
  -> repeatedly reload candidate vectors into QP working wavefunction arrays
  -> repeatedly build projected H q data through QP-side infrastructure
  -> save final Ritz vector back to EZFIO if requested
  -> print final energies and basis size

Python post-processing
  -> parse stdout
  -> extract per-iteration energies
  -> build CSV and plots
```

## What QP is giving CCIK that Python alone is not

In the maintained backend path, Quantum Package is not just a file format. It
provides:

- the EZFIO case model
- SCF and orbital setup
- determinant and wavefunction storage
- integral providers
- plugin configuration providers
- post-CCIK reuse of the final wavefunction

That is why the QP backend is a real runtime integration, not just "Python
calling a Fortran executable with some numbers."

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
