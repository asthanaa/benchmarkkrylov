# Quantum Package Backend Status

This note records the current Quantum Package backend state for baseline `ccik`.

## Scope

- Implemented in Quantum Package external plugin repo: `ccik_qp`
- Current algorithm coverage:
  - `ccik`: in progress, usable for development
  - `ccik_thick`: not ported
  - `ccik_stochastic`: not ported

## Layout

- Main repo: `CompressK`
  - validation and plotting
  - Python bridge script
  - experiment management
- Plugin repo: `ccik_qp`
  - Quantum Package external plugin
  - determinant-driven baseline `ccik`

## Local QP Environment

Quantum Package is bootstrapped under `/tmp/qp2`.

Working pieces:
- QP checkout and submodules
- configured build with `config/gfortran_macos.cfg`
- OCaml front-end tools rebuilt locally
- external plugin repo installed into QP
- direct plugin binary build:
  - `/tmp/qp2/plugins/ccik_qp/stable/ccik/ccik`

## Local Apple Silicon Fixes

These are local environment fixes in `/tmp/qp2`, not checked into this repo.

1. OCaml package fix
- patched the bundled opam/base stack to avoid invalid `-mpopcnt` on `arm64-darwin`

2. `qp_run.ml` compatibility fix
- replaced `String.starts_with` usage so it builds with OCaml `4.12`

3. EZFIO compressed-array fix
- patched `/tmp/qp2/ocaml/ezfio.ml` to use `gunzip -c` instead of `zcat`
- rebuilt:
  - `qp_create_ezfio.native`
  - `qp_set_mo_class.native`
  - `qp_run.native`
  - `qp_edit.native`

4. macOS memory-provider fix
- patched `/tmp/qp2/src/utils/memory.irp.f`
- avoids Linux-only `/proc/meminfo` access in direct binary runs

## Execution Mode

Use the compiled plugin binary directly.

Working path:
- `qp set_file <ezfio>`
- `qp set ccik ...`
- `/tmp/qp2/plugins/ccik_qp/stable/ccik/ccik <ezfio>`

Do not rely on `qp run ccik` on this machine.

Current issue:
- the OCaml/ZMQ wrapper still fails with:
  - `Context not closed before finalization`
  - `Socket not closed before finalization`
  - `Fatal error: exception Not_found`

Because of that, the bridge now treats direct-binary execution as the default.

## Plugin State

The plugin is no longer a stub.

Implemented pieces:
- plugin parameter providers in `EZFIO.cfg`
- sparse determinant-vector container in `ccik_sparse.f90`
- local generator/selector providers in `ccik_selection_providers.irp.f`
- baseline Krylov driver in `ccik_driver.irp.f`
- determinant-driven external discovery via QP `select_connected(...)`

Current baseline flow:
1. load current sparse Krylov vector into QP working wavefunction
2. define generators/selectors from the current support
3. build an external candidate buffer with `select_connected(...)`
4. evaluate projected `H q_k` amplitudes on:
   - current support
   - selected external determinants
5. orthogonalize/compress
6. solve the generalized Ritz problem using the actual overlap matrix

Recent correctness fixes in the plugin:
- solve generalized Ritz problem with `S = Q^T Q`, not plain `dsyev`
- deduplicate selection-buffer determinants before building projected `H q_k`

## N2 Development Check

System:
- N2
- `R = 1.10 Å`
- `6-31g`
- CAS(10,10)
- core orbitals: `1-2`
- active orbitals: `3-12`
- deleted orbitals: `13-18`

Reference from local PySCF CASCI:
- total energy: `-108.971450321154 Eh`

Observed QP-`ccik` behavior:

Poor budget:
- `m=4`
- `generator_keep=16`
- `selector_keep=64`
- `external_keep=128`
- `nkeep=128`
- total energy: `-115.169307073438 Eh`
- conclusion: projected matvec too crude; not usable

Working development budget:
- `m=4`
- `generator_keep=32`
- `selector_keep=256`
- `external_keep=256`
- `nkeep=256`
- total energy: `-108.970820613669 Eh`
- absolute error vs PySCF CASCI: `6.297e-04 Eh`

Larger budgets on the same point stayed in the same regime:
- `64/512/512/512`: `-108.970485563512 Eh`
- `128/1024/1024/1024`: `-108.970280737037 Eh`

Interpretation:
- the baseline QP port is numerically coherent on this N2 point
- small support budgets fail badly
- moderate support budgets already recover sub-millihartree accuracy at this geometry

## Short N2 Cross-Geometry Check

Tested with local PySCF CASCI references:

Moderate near-equilibrium setting:
- `ccik_m = 4`
- `ccik_generator_keep = 32`
- `ccik_selector_keep = 256`
- `ccik_external_keep = 256`
- `ccik_nkeep = 256`

Results:
- `R = 1.00 Å`: error `4.77e-04 Eh`
- `R = 1.10 Å`: error `6.30e-04 Eh`
- `R = 1.50 Å`: error `1.83e-02 Eh`
- `R = 2.50 Å`: error `1.59e-01 Eh`

Current stretched-safe development setting:
- `ccik_m = 16`
- `ccik_generator_keep = 512`
- `ccik_selector_keep = 8192`
- `ccik_external_keep = 8192`
- `ccik_nkeep = 8192`

Results:
- `R = 1.00 Å`: error `4.44e-07 Eh`
- `R = 1.10 Å`: error `5.81e-07 Eh`
- `R = 1.50 Å`: error `7.61e-06 Eh`
- `R = 2.00 Å`: error `3.91e-04 Eh`
- `R = 2.50 Å`: error `7.34e-04 Eh`

Conclusion:
- the current QP baseline port is already usable for N2 development with the stretched-safe setting
- the limiting factor is support growth at stretched bonds, not a broken environment or a broken single-point kernel
- recorded manual tuning data: `experiments/qp_ccik_n2_dev_20260312/scan_results.csv`
- recorded scripted development scan: `experiments/qp_ccik_n2_devcheck/results.csv`

## Bridge Script

Current bridge:
- `scripts/qp_ccik_bridge.py`

Default run mode:
- `direct`

Supported modes:
- `direct`
  - uses the compiled plugin binary under `$QP_ROOT/plugins/.../ccik`
- `qp-run`
  - keeps the old `qp run ccik` path available, but it is not the default

## Next Steps

1. Automate N2 scan setup through the direct-binary bridge.
2. Validate more N2 geometries with the current baseline port.
3. Find robust deterministic settings before extending to:
   - `ccik_thick`
   - `ccik_stochastic`
