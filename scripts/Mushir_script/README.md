# Mushir QRTE/QKUD 0.2 Plot Workflow

This folder contains the Talon submission workflow for a second set of
MolVQE-21 QRTE/QKUD plots at:

- QRTE `delta_t = 0.2`
- QKUD `epsilon = 0.2`

The workflow writes separate result directories and plot files so the original
`delta_t = 0.1` and `epsilon = 0.1` plots are not overwritten.

## Output Locations

Run outputs:

```text
runs/qrte_natorb_dt0.2
runs/qkud_natorb_eps0.2
```

Plot outputs:

```text
plots/mushir_dt0p2_eps0p2/adapt_vs_qrte_dt0p2_iterations_vs_entropy.csv
plots/mushir_dt0p2_eps0p2/adapt_vs_qrte_dt0p2_iterations_vs_entropy.png
plots/mushir_dt0p2_eps0p2/adapt_vs_qrte_dt0p2_iterations_vs_entropy.pdf
plots/mushir_dt0p2_eps0p2/adapt_vs_qkud_eps0p2_iterations_vs_entropy.csv
plots/mushir_dt0p2_eps0p2/adapt_vs_qkud_eps0p2_iterations_vs_entropy.png
plots/mushir_dt0p2_eps0p2/adapt_vs_qkud_eps0p2_iterations_vs_entropy.pdf
```

The CSV files use the same columns as the main plot tables. Cases with no
completed summary yet appear as `method_status=missing`; completed cases update
one at a time as the live watcher sees new `summary.json` files.

## Submit On Talon

From the repository root:

```bash
scripts/Mushir_script/submit_talon_dt0p2_eps0p2.sh
```

The submitter starts three jobs:

- QRTE pool job for `runs/qrte_natorb_dt0.2`
- QKUD pool job for `runs/qkud_natorb_eps0.2`
- live plot watcher for `plots/mushir_dt0p2_eps0p2`

The watcher starts immediately and refreshes whichever method has new completed
summaries. It exits after both watched Slurm jobs leave the queue and one final
refresh has run.

Useful overrides:

```bash
QCANT_N_STEPS=200 scripts/Mushir_script/submit_talon_dt0p2_eps0p2.sh
QCANT_CONCURRENCY=12 QCANT_MAX_WORKERS=3 scripts/Mushir_script/submit_talon_dt0p2_eps0p2.sh
```

## Manual Plot Refresh

Refresh both plots once:

```bash
python scripts/Mushir_script/collect_and_plot_dt0p2_eps0p2.py --method both
```

Watch for new summaries without submitting jobs:

```bash
python scripts/Mushir_script/watch_dt0p2_eps0p2_plots.py \
  --method both \
  --interval-seconds 60 \
  --allow-empty-refresh
```

## Renyi Plot With Mean-Field-Timestep QRTE And Original QKUD

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/plot_qrte_mean_field_trotter2_vs_qkud_eps0p05_renyi.py
```

This remakes the existing 27-system MolVQE Renyi comparison from saved data.
QRTE comes from the small10 and remaining17 mean-field active-window runs:
`dt=pi/W_MF`, second-order Trotter, one slice. It keeps the original QKUD
`epsilon=0.05` points, corrected/Fe rerun selections, ADAPT background, and
manifest `h0.25` values. All 27 saved QKUD configurations use Trotter mode and
one slice; the QKUD circuit builder uses its default first-order formula.

Outputs share the prefix
`plots/qkud_eps0p05_vs_qrte_mean_field_pi_over_w_trotter2_steps1_iterations_vs_entropy`
with `.png`, `.pdf`, `.csv`, and `.json` extensions. The script validates QRTE
settings, Hamiltonian/source-spectrum hashes, and history-derived hit iterations.
It plots 25 QRTE hits and 27 QKUD hits; FeO and FeNO2+ did not reach chemical
accuracy within 100 QRTE iterations. Their QKUD hits are yellow stars, and
their QRTE hit iterations stay blank in the CSV. Fits use successful cases
only, with R-squared measured in log-iteration space.

Legacy QKUD accuracy references are retained, including the differing CO spin 2
and Al3-minus targets; the CSV records their differences from the QRTE references.
The JSON records source and plotting conventions. No molecular calculations
are launched, and the earlier fixed-timestep plot is preserved.

## QRTE With A Mean-Field Timestep

Submit the validated 27-system MolVQE set with a separate timestep for each
system:

```bash
bash scripts/Mushir_script/submit_talon_qrte_fock_pi_over_w.sh
```

Each array task first runs the original mean-field reference and sets
`delta_t = pi / (max(mf.mo_energy) - min(mf.mo_energy))`. This uses **all SCF
orbitals**, including core and virtual orbitals. These are one-electron orbital
energies, not the total HF energy, many-electron determinant energies, or the
interacting active-space spectrum. For ROHF and sfX2C1eROHF it uses PySCF's
effective Roothaan orbital energies. The mean-field calculation has no
active-space restriction.

QRTE then loads the exact Hamiltonian selected in
`hamiltonian_cache/correct_27_systems/manifest.csv`, checked against its recorded
SHA-256. Cyclobutadiene and both m-benzyne spin cases retain the corrected
orbital indices. `feno_2plus` and `fenoco3_minus` retain sfX2C1eROHF references;
FeO retains its validated ROHF reference and Bohr geometry. `fenoco3_minus`
retains the previously used final unconverged CASSCF/natorb cache, with that
status recorded explicitly; its new SCF calculation must converge.

Ferrocene is excluded. `feo4_2minus_ceo` is also outside this validated 27-system
set because the corrected reference attempt failed; no stale cache is substituted.
The SCF step preserves the earlier settings and checks its energy against the
cached SCF energy where available. It does not rerun CASSCF or replace any
Hamiltonian caches.

Defaults: 100 Krylov iterations, early stopping at 0.0016 Eh chemical accuracy,
overlap tolerance 1e-10, Trotter evolution with one step, and up to four concurrent
Slurm array tasks, with a 12-hour default wall-time limit. The accuracy target comes from the Hamiltonian cache's own
CASSCF energy. This matters for CO spin 2 and Al3-minus, whose previous summaries
used different dataset target energies. Using the full Fock orbital bandwidth
does not give an aliasing bound for the interacting QRTE Hamiltonian.

Results go to `runs/qrte_fock_pi_over_w/<case_id>/`: `fock_scf.log`,
`fock_reference.json` (full orbital spectrum), `run_config.json` (W and delta_t),
`input_snapshot.json`, `min_energy_history.csv`, and `summary.json`. Restarting
skips completed QRTE runs with matching settings and reuses matching SCF results.

```bash
# Validate all cache mappings and references without SCF or QRTE calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_fock_pi_over_w.py --dry-run

# Compute only the SCF spectra and timesteps on the cluster.
QCANT_REFERENCE_ONLY=1 bash scripts/Mushir_script/submit_talon_qrte_fock_pi_over_w.sh

# Optional run settings. Extra arguments are passed to sbatch.
QCANT_N_STEPS=100 QCANT_OVERLAP_TOL=1e-9 QCANT_ARRAY_CONCURRENCY=2 \
  bash scripts/Mushir_script/submit_talon_qrte_fock_pi_over_w.sh --time=48:00:00
```

Other environment overrides: `PYTHON`, `QCANT_OUTDIR`, `QCANT_CACHE_MANIFEST`,
`QCANT_MAX_WORKERS`, `QCANT_EVOLUTION_MODE`, `QCANT_TROTTER_STEPS`,
`QCANT_SCF_MATCH_TOLERANCE`, and `QCANT_FORCE=1`. Use separate output directories
for different run settings when both sets of results are needed.

## Ten Small Systems With A UHF Determinant Bandwidth

```bash
bash scripts/Mushir_script/submit_talon_qrte_uhf_determinant_pi_over_w_small10.sh
```

The fixed array order is H4/STO-3G spin 0, H5/STO-3G spin 1, H6/STO-3G spin 0,
H6/cc-pVTZ spin 0, BeH2/cc-pVTZ spins 0 and 2, N2/cc-pVTZ spins 0 and 2, and
CO/cc-pVTZ spins 0 and 2. Exact case IDs are in `CASE_IDS` in
`run_qrte_uhf_determinant_pi_over_w_small10.py`. Each QRTE space has at most six
active orbitals.

Every task runs a full-system UHF calculation at the original geometry, charge,
spin, and basis. At fixed **total molecular** Nalpha and Nbeta, the lowest
frozen-Fock determinant occupies the Nalpha lowest alpha orbitals and Nbeta
lowest beta orbitals. The highest occupies the corresponding highest orbitals.
`W` is the difference between their summed orbital energies and `delta_t=pi/W`.
Core and virtual orbitals are included; neither the orbital energies nor the
electron counts are restricted to the QRTE active space.

This is the determinant spectrum of a single converged UHF Fock operator.
Excited determinants are not optimized in separate SCF calculations, and the
orbital sums are not interacting-Hamiltonian determinant expectation values.
A common energy offset matches the occupied reference determinant to the total
UHF energy; that offset cancels from W. The saved reference includes alpha/beta
orbital energies, extremal determinant occupations, total UHF energy, and S^2.
UHF may converge to a spin-restricted solution; this workflow does not perform
a separate stability search or force spin symmetry breaking.

Only the timestep source changes. QRTE uses the same exact Hamiltonian caches,
initial determinants, CASSCF accuracy targets, 100-iteration limit, chemical
accuracy stopping, overlap tolerance 1e-10, and one-step Trotter evolution as the
previous workflow. UHF energies are not required to match RHF/ROHF energies.
The array uses the current submission settings: 36 CPUs, 8G memory, two days,
and four concurrent tasks.

Outputs go to `runs/qrte_uhf_determinant_pi_over_w_small10/<case_id>/`, using
the same file names as the earlier workflow. `fock_reference.json` contains the
UHF results and `run_config.json` records the determinant W and timestep.

```bash
# Check inputs without molecular calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_uhf_determinant_pi_over_w_small10.py --dry-run

# Optional: run only UHF and save determinant bandwidths/timesteps.
QCANT_REFERENCE_ONLY=1 \
  bash scripts/Mushir_script/submit_talon_qrte_uhf_determinant_pi_over_w_small10.sh
```

The submitter accepts the same run and Slurm environment overrides as the
previous workflow, except `QCANT_SCF_MATCH_TOLERANCE` does not apply to the UHF
source of W. Extra command-line arguments are passed to `sbatch`.

## Same Ten Systems With A CASSCF Natural-Orbital Bandwidth

```bash
bash scripts/Mushir_script/submit_talon_qrte_casscf_natorb_pi_over_w_small10.sh
```

This uses the identical ten cases and array order from the UHF experiment.
Each task reads the exact validated CASSCF/natorb Hamiltonian cache and builds
the **complete interacting active-space CI Hamiltonian** at the cache's active
Nalpha and Nbeta. PySCF constructs the full sector matrix, and SciPy diagonalizes
it to obtain E0 and Emax. `W=Emax-E0` and `delta_t=pi/W`, in atomic units.
The spectrum includes the one- and two-electron interactions and the common
core energy. The sector fixes N and Sz without additional total-spin or
spatial-irrep projection. It uses the active-space electron counts and orbitals.

No new SCF or CASSCF optimization is performed: the existing optimized
natural-orbital integrals are reused. The script checks that E0 matches the
cache's CASSCF energy within 1e-6 Eh and fails before QRTE if it does not.
The same cache then supplies QRTE, with the same initial determinant, reference
energy, 100-iteration limit, chemical accuracy stopping, overlap tolerance
1e-10, and one-step Trotter evolution. The Slurm resources match the UHF run.

Outputs are in `runs/qrte_casscf_natorb_pi_over_w_small10/<case_id>/`:
`active_space_reference.json` contains all sector eigenvalues, E0, Emax, the
sector dimension, and the reference-energy check. `run_config.json` and
`summary.json` record `active_space_bandwidth_hartree` and `delta_t`.
`min_energy_history.csv` contains QRTE convergence; `input_snapshot.json`
records cache and molecular provenance. Slurm logs use `mushir-qrte-cas-piW-*`.

```bash
# Input/cache checks only; no diagonalization or QRTE.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10.py --dry-run

# Optional: diagonalize the cached active spaces and save timesteps only.
QCANT_REFERENCE_ONLY=1 \
  bash scripts/Mushir_script/submit_talon_qrte_casscf_natorb_pi_over_w_small10.sh
```

Run and Slurm environment overrides work as in the UHF workflow. The fixed
CASSCF reference-energy match tolerance is independent of the old
`QCANT_SCF_MATCH_TOLERANCE` setting.

## Same Ten Systems With Exact QRTE Propagation

```bash
bash scripts/Mushir_script/submit_talon_qrte_casscf_natorb_pi_over_w_small10_exact.sh
```

The dedicated exact array reuses `run_qrte_casscf_natorb_pi_over_w_small10.py`
and the same validated ten Hamiltonian caches. It sets `evolution_mode=sparse`:
SciPy `expm_multiply` applies the Hamiltonian exponential to numerical precision,
without Trotterization. The exact interacting CAS width, `delta_t=pi/W`, initial
determinants, reference energies, overlap tolerance 1e-10, 100-iteration limit,
and chemical accuracy stopping at 1.6 mHa remain unchanged. `trotter_steps` is
ignored in sparse mode. Slurm resources remain 36 CPUs, 8G, two days, and four
concurrent tasks; this implementation constructs the full qubit-space sparse
Hamiltonian, rather than propagating directly in the smaller determinant sector.

Outputs go to `runs/qrte_casscf_natorb_pi_over_w_small10_exact/<case_id>/`,
leaving the previous Trotterized results untouched. File names are the same as
in the CAS-width workflow. Slurm logs use `mushir-qrte-cas-piW-exact-*`; the job
ID is saved to
`provenance/latest_qrte_casscf_natorb_pi_over_w_small10_exact_jobid.txt`.
The usual run and Slurm overrides are supported, except evolution mode is
always `sparse`. Use a new `QCANT_OUTDIR` for each comparison to preserve results.

```bash
# Validate inputs and exact-mode arguments without diagonalization or QRTE.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10.py \
  --evolution-mode sparse \
  --outdir runs/qrte_casscf_natorb_pi_over_w_small10_exact --dry-run
```

## Same Ten Systems With Second-Order Trotter Propagation

```bash
bash scripts/Mushir_script/submit_talon_qrte_casscf_natorb_pi_over_w_small10_trotter2.sh
```

This uses the same ten validated CASSCF natural-orbital caches and exact
interacting active-space width, `W=Emax-E0`, with `delta_t=pi/W`. Only the
propagation formula changes: the Qulacs circuit uses a forward sweep of
half-time Pauli rotations followed by a reverse sweep of half-time rotations
(symmetric second-order Suzuki-Trotter). The default is one slice per QRTE
timestep. Order 2 is not the same as two first-order slices. Initial determinants,
reference energies, the 100-iteration limit, overlap tolerance 1e-10, and
chemical-accuracy stopping at 1.6 mHa remain unchanged. Slurm resources remain
36 CPUs, 8G, two days, and four concurrent tasks.

Outputs go to `runs/qrte_casscf_natorb_pi_over_w_small10_trotter2/<case_id>/`.
Existing first-order and exact outputs are preserved. The saved configuration
and summary record `trotter_order=2` and `trotter_steps=1`. Logs use
`mushir-qrte-cas-piW-trotter2-*`; the latest job ID is recorded in
`provenance/latest_qrte_casscf_natorb_pi_over_w_small10_trotter2_jobid.txt`.

```bash
# Input/cache checks only, without diagonalization or QRTE.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10.py \
  --evolution-mode trotter --trotter-order 2 --trotter-steps 1 \
  --outdir runs/qrte_casscf_natorb_pi_over_w_small10_trotter2 --dry-run

# Optional: use two second-order slices, keeping the total timestep unchanged.
QCANT_TROTTER_STEPS=2 \
QCANT_OUTDIR=runs/qrte_casscf_natorb_pi_over_w_small10_trotter2_steps2 \
  bash scripts/Mushir_script/submit_talon_qrte_casscf_natorb_pi_over_w_small10_trotter2.sh
```

The dedicated wrapper pins evolution mode to `trotter` and order to 2.
Other run and Slurm overrides work as in the CAS-width workflow. First-order
remains the shared driver's default, and existing first-order resume profiles
remain compatible. Second-order profiles are distinct to prevent incorrect
reuse of completed runs.

## Same Ten Systems With Second Order And Fixed dt=0.05

```bash
bash scripts/Mushir_script/submit_talon_qrte_casscf_natorb_small10_trotter2_dt0p05.sh
```

This uses the same ten validated CASSCF/natorb Hamiltonian caches and initial
determinants, but fixes `delta_t=0.05` atomic units for every case. The Qulacs
propagator uses symmetric second-order Trotterization with one slice per QRTE
timestep by default. It retains the 100-iteration limit, overlap tolerance
1e-10, and chemical-accuracy stopping at 1.6 mHa against the same references.
Slurm resources are unchanged: 36 CPUs, 8G, two days, four concurrent tasks.

Outputs go to `runs/qrte_casscf_natorb_small10_trotter2_dt0p05/<case_id>/`.
Earlier results remain untouched. The existing CAS spectrum calculation still
records W for comparison, but W does not set the timestep in this run.
`run_config.json` and `summary.json` record `delta_t=0.05`, the fixed-timestep
rule, and the unused bandwidth-derived timestep as `bandwidth_delta_t`.
The fixed timestep is part of the resume profile, so a different timestep
cannot reuse completed QRTE results. Logs use `mushir-qrte-cas-trotter2-dt005-*`;
the job ID is saved to
`provenance/latest_qrte_casscf_natorb_small10_trotter2_dt0p05_jobid.txt`.

```bash
# Validate inputs without diagonalizing or running QRTE.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10.py \
  --evolution-mode trotter --trotter-order 2 --trotter-steps 1 --delta-t 0.05 \
  --outdir runs/qrte_casscf_natorb_small10_trotter2_dt0p05 --dry-run
```

The dedicated scripts pin order 2, Trotter evolution, and dt=0.05. Other run
and Slurm overrides work as before; use a distinct `QCANT_OUTDIR` when changing
the number of slices. The shared driver also accepts `--delta-t` directly,
while its default remains `pi/W` when that argument is absent.

## Active-Orbital Mean-Field Width Comparison

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/compare_active_fock_width_small10.py
```

This analysis uses the saved ten-system HF orbital spectra and exact CAS
widths; it launches no SCF, CASSCF, CI, or QRTE runs. It writes
`runs/active_fock_width_small10/comparison.csv` and `comparison.json`.
The first estimate uses a canonical molecular HF window after the frozen-core
orbitals, or recorded explicit indices when available. It compares the maximum
minus minimum fixed-active-Nalpha/Nbeta sums of orbital energies with the
interacting CAS width. The one-electron range is also recorded separately.
ROHF uses its saved effective Roothaan spatial spectrum, not separate alpha/beta
Fock spectra. A canonical window is not guaranteed to reproduce the optimized
CASSCF active subspace; CO's AVAS projector is not reconstructed by this check.

A second estimate builds physical alpha/beta Fock matrices from the frozen
QRTE reference determinant and cached natural-orbital CAS integrals using
PySCF Coulomb/exchange contractions. This preserves the actual CAS subspace
but is neither the original molecular SCF calculation nor a new self-consistent
HF optimization. Only these small one-electron matrices are diagonalized.
Neither estimate optimizes the highest-energy HF determinant, and frozen Fock
orbital sums are not interacting determinant energies. The JSON includes the
orbital energies, selected indices, provenance, caveats, width ratios, and
implied timestep ratios for both methods. No existing run outputs are changed.

## Same Ten Systems With Mean-Field Active-Window W And Second Order

```bash
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_pi_over_w_small10_trotter2.sh
```

This uses the first mean-field estimate above: the saved original molecular
RHF/ROHF orbital energies in an active-sized canonical window after the core,
or recorded initial active indices when present. For each spin, sum the highest
and lowest `Nalpha`/`Nbeta` selected orbital energies; their total difference is
`W_MF`. The timestep is `pi/W_MF`, not pi divided by the single-orbital range.
No new SCF calculation or interacting CAS diagonalization is needed.

Only the timestep source changes relative to the second-order CAS-width runs.
QRTE retains the validated CASSCF/natural-orbital Hamiltonian caches and initial
determinants, symmetric second-order Qulacs propagation with one slice per
timestep, at most 100 iterations, overlap tolerance 1e-10, and chemical-accuracy
stopping at 1.6 mHa. Slurm retains 36 CPUs, 8G, two days, and four concurrent tasks.
The dedicated driver defaults to second order and rejects fixed-time, sparse,
or first-order overrides. The batch script also pins the mode and order;
inherited `QCANT_DELTA_T` is not used.

Outputs: `runs/qrte_mean_field_active_pi_over_w_small10_trotter2/<case_id>/`.
Each run saves `mean_field_active_reference.json`, `run_config.json`,
`input_snapshot.json`, `min_energy_history.csv`, and `summary.json`.
The selected HF indices, occupations, energies, determinant sums, W, timestep,
and approximation caveats are recorded. Source spectra are read from
`runs/qrte_fock_pi_over_w/<case_id>/fock_reference.json`; their convergence,
method, geometry, active selection, and Hamiltonian provenance are checked.
Their SHA256 hashes are included in resume profiles so changed spectra cannot
reuse completed results. Previous run directories are left unchanged.
Logs use `mushir-qrte-mf-active-piW-trotter2-*`; the job ID is recorded in
`provenance/latest_qrte_mean_field_active_pi_over_w_small10_trotter2_jobid.txt`.

The canonical HF window is not generally the optimized CASSCF subspace and
does not reconstruct CO's AVAS projector. ROHF uses the saved effective
Roothaan spatial energies for both spin channels. These are frozen-Fock
determinant orbital sums, not relaxed extremal HF determinant energies.
The actual QRTE Hamiltonians still retain the original active-orbital fixes.

```bash
# Validate saved inputs and display W/dt; no molecular calculations or QRTE.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_pi_over_w_small10.py --dry-run
```

Run and resource overrides work as before (`QCANT_N_STEPS`,
`QCANT_OVERLAP_TOL`, `QCANT_TROTTER_STEPS`, `QCANT_MAX_WORKERS`,
`QCANT_ARRAY_CONCURRENCY`, `QCANT_OUTDIR`, `QCANT_REFERENCE_ONLY`,
`QCANT_FORCE`, and additional `sbatch` arguments). Use a distinct output
directory when changing the number of slices.

## Same Ten Systems With Mean-Field W, First Order, And Four Slices

Prepared submission command:

```bash
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4.sh
```

This uses the same ten cases, saved RHF/ROHF spectra, active-window determinant
bandwidth `W_MF`, and full timestep `delta_t=pi/W_MF` as the second-order
mean-field workflow. The Qulacs circuit applies **four first-order Trotter
slices per timestep**, each of duration `delta_t/4`. Four slices do not change
the total timestep or the interpretation of one QRTE iteration.

Validated CASSCF/natural-orbital caches, original QRTE initial determinants,
reference energies, 100-iteration cap, overlap tolerance 1e-10, and stopping
at 1.6 mHa are retained. No SCF, CASSCF, or interacting CAS diagonalization is
needed. The dedicated driver and batch script require Trotter mode, order 1,
and four slices. Inherited `QCANT_TROTTER_ORDER`, `QCANT_TROTTER_STEPS`,
`QCANT_EVOLUTION_MODE`, and `QCANT_DELTA_T` do not change this experiment.

Outputs go to
`runs/qrte_mean_field_active_pi_over_w_small10_trotter1_steps4/<case_id>/`,
with the same saved-reference, configuration, input, history, and summary
files as the earlier mean-field run. Resume profiles distinguish propagation
settings and saved-spectrum hashes. Logs use
`mushir-qrte-mf-active-piW-trotter1-steps4-*`; the separate job record is
`provenance/latest_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4_jobid.txt`.
Slurm defaults are 36 CPUs, 8G, **6 hours**, and four concurrent array tasks.

```bash
# Validate all ten saved spectra and caches, without molecular calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4.py --dry-run
```

Other run/resource overrides work as before (`QCANT_N_STEPS`,
`QCANT_OVERLAP_TOL`, `QCANT_MAX_WORKERS`, `QCANT_ARRAY_CONCURRENCY`,
`QCANT_OUTDIR`, `QCANT_REFERENCE_ONLY`, `QCANT_FORCE`, and additional
`sbatch` arguments). Preparation does not submit jobs.

## CO Spin 2 At dt=0.05 With A Matching Cached Reference

The older `runs/qrte_natorb_dt0.05/CO_equi_cc_pVTZ_2` run used the dataset
manifest's CASSCF energy as its stopping target while constructing the local
CASSCF/natorb Hamiltonian separately. `scripts/build_manifest.py` copies
`meta.casscf.energy` directly from the dataset JSON, and the original
`scripts/run_krylov_benchmark.py` uses that manifest value without checking
it against the Hamiltonian cache's own reference. Its one-sided hit test also
accepts energies below the dataset target. For this CO case, the two references
differ, so the old run stopped after one iteration before reaching the ground
energy of its local cached Hamiltonian.

| Quantity | Energy (Eh) |
| --- | ---: |
| MolVQE JSON `meta.casscf.energy` | -112.60166830789687 |
| Validated cached CASSCF reference | -112.65639016965865 |
| Saved exact fixed-Nalpha/Nbeta sector ground energy | -112.65639016965986 |

The dataset-minus-cache difference is 54.72186176 mHa. The exact cached-sector
ground energy agrees with the cached reference to about 1.2e-12 Eh. The
dataset and local SCF energies agree to about 2.2e-11 Eh; the discrepancy appears
at the CASSCF/active-space stage. Saved evidence does not establish the precise
orbital or state-selection reason. The dataset energy is not overwritten or
declared erroneous: the cached reference is the matching target for the local
Hamiltonian used in these QRTE runs. This is not an exact reproduction of the
dataset's CO CASSCF reference.

Prepared CO-only rerun (submit only when calculations are requested):

```bash
bash scripts/Mushir_script/submit_talon_qrte_co_spin2_trotter1_dt0p05.sh
```

It pins `dt=0.05`, first-order Trotter, one slice, the validated Hamiltonian,
and the cached reference. Defaults are 100 iterations, overlap tolerance
1e-10, chemical accuracy 1.6 mHa, 36 CPUs, 8G, and **6 hours**. It reuses the
saved exact CAS spectrum to validate the reference without new SCF, CASSCF,
or diagonalization. Configuration and histories record both reference energies;
hit flags use absolute error against the cached reference. The older run is
preserved. Outputs go to
`runs/qrte_co_spin2_trotter1_dt0p05_correct_reference/CO_equi_cc_pVTZ_2/`.
Its separate job record is
`provenance/latest_qrte_co_spin2_trotter1_dt0p05_jobid.txt`.

```bash
# Validate the cache, dataset entry, saved exact spectrum, and settings only.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_co_spin2_trotter1_dt0p05.py --dry-run
```

## Four-System QRTE Timestep Sensitivity Scan

The scan uses N2/cc-pVTZ spins 0 and 2, Al3-minus, and H10/STO-3G spin 0.
For each system, `dt0=pi/W_MF` uses the same saved original molecular HF
active-window determinant bandwidth as the earlier mean-field runs. The fixed
factors are **0.25, 0.5, 0.75, 1, 1.25, 1.5, 2**, giving 28 conditions.
Each full QRTE timestep is `dt=factor*dt0`; propagation is symmetric
second-order Trotter with exactly one slice. Changing the factor does not
change the Hamiltonian, initial determinant, or reference energy.

| Array tasks | Case ID | dt0 (a.u.) | Scan range (a.u.) | Earlier dt0 hit iteration |
| --- | --- | ---: | ---: | ---: |
| 0–6 | `N2_equi_cc_pVTZ_0` | 0.653130255 | 0.163282564–1.306260510 | 5 |
| 7–13 | `N2_equi_cc_pVTZ_2` | 0.932280660 | 0.233070165–1.864561321 | 7 |
| 14–20 | `Al3_minus` | 1.224443715 | 0.306110929–2.448887430 | 4 |
| 21–27 | `H10_near_equi_sto3g_0` | 0.267028035 | 0.066757009–0.534056069 | 6 |

The task order is system first, then the seven factors in the order above.
All 28 conditions have dedicated new run directories, including the factor-1
points; earlier baseline runs are preserved. Defaults remain 100 iterations,
overlap tolerance 1e-10, and chemical accuracy 1.6 mHa against each validated
cache's own CASSCF/natorb reference. This includes Al3-minus's cached reference,
rather than its slightly different dataset energy. No new SCF, CASSCF, or
interacting CAS diagonalization is needed.

```bash
# Read-only validation and display of all 28 conditions.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_dt_scan4.py --dry-run

# Prepare scan_plan.csv/.json without launching calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan4.py --prepare-only

# Submit only when calculations are explicitly requested.
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_dt_scan4.sh
```

Slurm defaults are **6 hours per task**, 36 CPUs, 8G, and four concurrent
tasks in a 0–27 array. The wrapper checks all inputs and records the scan plan
before `sbatch`. Inherited fixed-time, order, and slice overrides cannot alter
the experiment. Standard overrides include `QCANT_N_STEPS`, `QCANT_OVERLAP_TOL`,
`QCANT_MAX_WORKERS`, `QCANT_ARRAY_CONCURRENCY`, `QCANT_OUTDIR`,
`QCANT_CACHE_MANIFEST`, `QCANT_REFERENCE_ONLY`, `QCANT_FORCE`, and extra
`sbatch` arguments. A changed scan plan requires a separate output root.

Outputs go to
`runs/qrte_mean_field_active_dt_scan4_trotter2/factor_<scale>/<case_id>/`,
for example `factor_0p25/N2_equi_cc_pVTZ_0/`. Each run saves the usual mean-field
reference, input, configuration, history, and summary; configuration/summary
also record the factor and unscaled `pi/W_MF` timestep. Resume profiles include
the factor, propagation settings, and saved HF source hash. Logs use
`mushir-qrte-mf-dt-scan4-trotter2-*`; the separate job record is
`provenance/latest_qrte_mean_field_active_dt_scan4_jobid.txt`.

After or during the scan, collect saved results without running calculations:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan4.py
```

The collector writes `scan_results.csv` and `iterations_vs_dt_scale.png/.pdf`
under the scan root, with one panel per system. It verifies result settings,
Hamiltonian/HF source hashes, and history-derived chemical-accuracy hits.
Pending, failed, invalid, and completed no-hit conditions remain distinct.
A completed no-hit point gets a hollow triangle at the iteration cap and a
blank hit iteration in the CSV; it is not reported as a 100-iteration success.
When using nondefault iteration, overlap, worker, or output settings, pass the
same values to the collector (`--n-steps`, `--overlap-tol`, `--max-workers`,
`--outdir`).

## BeH2 And H4/H6/H7 Timestep Scan

This separate scan reuses the same mean-field width, factors
**0.25, 0.5, 0.75, 1, 1.25, 1.5, 2**, second-order Trotter, and one slice.
There are seven validated cases and **49 conditions**. H6 includes both bases
and the two validated cc-pVTZ spin cases:

| Array tasks | Case ID | dt0=pi/W_MF (a.u.) |
| --- | --- | ---: |
| 0–6 | `BeH2_equi_cc_pVTZ_0` | 1.327096280 |
| 7–13 | `BeH2_equi_cc_pVTZ_2` | 1.717978885 |
| 14–20 | `H4_near_equi_sto3g_0` | 0.724456799 |
| 21–27 | `H6_near_equi_sto3g_0` | 0.461237356 |
| 28–34 | `H6_near_equi_cc_pVTZ_0` | 0.699023035 |
| 35–41 | `H6_near_equi_cc_pVTZ_2` | 0.887815142 |
| 42–48 | `H7_near_equi_sto3g_1` | 0.411091662 |

Within each block, factors follow the order listed above. Full timestep is
`factor*dt0`. Hamiltonian caches, initial determinants, CASSCF reference energies,
the 100-iteration cap, overlap tolerance 1e-10, and 1.6 mHa chemical-accuracy
stopping are retained. No new SCF, CASSCF, or CAS diagonalization is needed.
The factor-1 points have separate output directories and do not overwrite
earlier centre-point results.

```bash
# Read-only validation of all 49 tasks.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_dt_scan_beh2_hchains.py --dry-run

# Write scan_plan.csv/.json without running calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan_beh2_hchains.py --prepare-only

# Submit only when calculations are explicitly requested.
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_dt_scan_beh2_hchains.sh
```

Slurm requests **6 hours per task**, 36 CPUs, 8G, and four concurrent tasks in
an array 0–48. Standard environment and Slurm overrides work as for scan4;
propagation stays pinned to order 2 and one slice. The output root is
`runs/qrte_mean_field_active_dt_scan_beh2_hchains_trotter2`, with runs under
`factor_<scale>/<case_id>/`. A different plan or source spectrum requires a
separate output root. Logs use `mushir-qrte-mf-dt-scan-beh2-hchains-trotter2-*`,
and the separate job record is
`provenance/latest_qrte_mean_field_active_dt_scan_beh2_hchains_jobid.txt`.

Collect results and create a seven-panel iteration-versus-timestep plot with:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan_beh2_hchains.py
```

This writes `scan_results.csv` and `iterations_vs_dt_scale.png/.pdf` in the
new scan root. History-derived hit validation and explicit pending, failed,
invalid, and completed no-hit statuses are identical to the four-system scan.
The collector never launches molecular calculations.

## Exact Exponential Versions Of Both Timestep Scans

The exact scans repeat all **11 cases and 77 conditions** from the two scans
above. The four-case group contains N2/cc-pVTZ spins 0/2, Al3-minus, and
H10/STO-3G spin 0. The seven-case group contains BeH2/cc-pVTZ spins 0/2,
H4/STO-3G spin 0, H6/STO-3G spin 0, H6/cc-pVTZ spins 0/2, and H7/STO-3G spin 1.
Case order, factors **0.25, 0.5, 0.75, 1, 1.25, 1.5, 2**, saved HF widths,
`dt=factor*pi/W_MF`, cached interacting Hamiltonians, initial determinants,
and cached CASSCF reference energies match the previous Trotter scans.

Propagation uses `evolution_mode=sparse`: the existing QRTE implementation
applies `expm_multiply(-1j * dt * H, state)` without a Trotter product formula.
The mean-field width remains the timestep estimate; it is not replaced by
an interacting CAS width. No new SCF, CASSCF, or CAS spectrum is calculated.
The API's order/slice values `(1, 1)` are unused in sparse mode; the exact
scan plans explicitly record `trotter_parameters_apply=false`.

Defaults remain 100 QRTE iterations, overlap tolerance `1e-10`, and
chemical-accuracy stopping at 1.6 mHa against the cached reference. Each exact
array requests 48 hours per task, 36 CPUs, and at most four concurrent tasks.
The four-case array requests 64G per task for the larger Al3-minus (18-qubit)
and H10 (20-qubit) sparse Hamiltonians; the seven-case array requests 8G.
These are resource requests, not measured runtime or peak-memory guarantees.
The existing scan environment/Slurm overrides also apply to these wrappers;
the evolution mode and scaled timestep rule are pinned by the exact drivers.

```bash
# Validate the 28- and 49-condition grids without running QRTE.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_dt_scan4_exact.py --dry-run
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_dt_scan_beh2_hchains_exact.py --dry-run

# Prepare scan_plan.csv/.json without launching calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan4_exact.py --prepare-only
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan_beh2_hchains_exact.py --prepare-only

# Submit the two exact arrays when calculations are requested.
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_dt_scan4_exact.sh
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_dt_scan_beh2_hchains_exact.sh
```

The new roots are `runs/qrte_mean_field_active_dt_scan4_exact` and
`runs/qrte_mean_field_active_dt_scan_beh2_hchains_exact`, each with
`factor_<scale>/<case_id>/` subdirectories. Their CSV/JSON scan plans include
the timestep, reference, Hamiltonian hash, and saved mean-field source hash
for every condition. Exact outputs, logs, and latest-job records have
separate names. A changed scan plan requires a separate output root.

Collect results by running the corresponding `collect_*_exact.py` command
without `--prepare-only`; add `--loglog` for the logarithmic plot variant.
Collectors validate exact propagation, source hashes, reference energies,
and history-derived hits. Pending, failed, invalid, and completed no-hit
conditions remain distinct. Trotterized results cannot count as exact hits.

## Exact Timestep Scan For Systems Up To 12 Qubits

`run_qrte_mean_field_active_dt_scan_upto12_exact.py` combines the eight small
cases into a separate scan with **88 conditions**, indexed 0–87. The factors
are **0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2**, with
`dt=factor*pi/W_MF`. All earlier seven factors are included, with four new
points near zero. The Hamiltonian, initial determinant, saved HF width,
cached reference, 100-iteration limit, overlap tolerance `1e-10`, and 1.6 mHa
stopping rule retain the earlier exact-scan settings.

| Task indices | Case ID | Qubits |
| --- | --- | ---: |
| 0–10 | `N2_equi_cc_pVTZ_0` | 12 |
| 11–21 | `N2_equi_cc_pVTZ_2` | 12 |
| 22–32 | `BeH2_equi_cc_pVTZ_0` | 12 |
| 33–43 | `BeH2_equi_cc_pVTZ_2` | 12 |
| 44–54 | `H4_near_equi_sto3g_0` | 8 |
| 55–65 | `H6_near_equi_sto3g_0` | 12 |
| 66–76 | `H6_near_equi_cc_pVTZ_0` | 12 |
| 77–87 | `H6_near_equi_cc_pVTZ_2` | 12 |

The driver verifies the 12-qubit limit against the loaded active spaces.
Exact propagation uses sparse exponential action. The launcher submits **four
worker jobs**, each processing a strided subset of the 88 conditions sequentially
(`task_index % worker_count == worker_index`). Each worker requests 36 CPUs,
8G and 48 hours. Outputs use
`runs/qrte_mean_field_active_dt_scan_upto12_exact/factor_<scale>/<case_id>/`.
The 88 scientific task indices and output paths remain unchanged. Slurm array
indices identify workers, so the default array is `0-3%4`, with 22 conditions
assigned to each worker.

This avoids `AssocMaxSubmitJobLimit`: the account permits 50 submitted jobs
and 6 running jobs, and an 88-element array counts as 88 submitted jobs even
with `%4`. `QCANT_SCAN_WORKERS` adjusts the number of worker jobs, while
`QCANT_ARRAY_CONCURRENCY` caps how many workers run simultaneously. Restarting
the launcher skips completed compatible conditions using the existing resume
checks. Worker failures are reported after attempting their remaining conditions.

```bash
# Validate all 88 conditions without calculating or submitting anything.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_dt_scan_upto12_exact.py --dry-run

# Prepare plans and copy compatible completed runs from the earlier exact scans.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan_upto12_exact.py --prepare-only

# Submit the dedicated array when calculations are requested.
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_dt_scan_upto12_exact.sh
```

Preparation reuses only completed exact runs with the same full run profile,
timestep, Hamiltonian/HF hashes, cached reference and history-derived hit.
Each copied run records its original source and copied-file hashes in
`reused_from.json`; existing destinations and original results are preserved.
At initial preparation, 14 N2 runs are reused and 74 conditions remain pending.
These reused tasks resume as completed when a worker reaches them.

Run the collector without `--prepare-only` to write `scan_results.csv` and
the eight-panel plot. Add `--loglog` to resolve the new near-zero factors.
No molecular calculations or Slurm jobs are launched by the collector.

Generate separate publication-quality log-log plots for all eight systems:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/plot_qrte_mean_field_dt_scan_upto12_exact_loglog.py
```

The script validates the saved plan, Hamiltonian/HF hashes, configurations,
and history-derived hit iterations. It writes individual PDF/600-dpi PNG/CSV
files and `all_systems_loglog.pdf/.png` under
`runs/qrte_mean_field_active_dt_scan_upto12_exact/loglog_plots/`. The x-axis
is the logarithmically scaled factor `dt/(pi/W_MF)` and the y-axis is the
logarithmically scaled chemical-accuracy hit iteration. Completed no-hit
conditions appear as open triangles with upward arrows at their completed
iteration counts; their hit iterations remain blank in the CSVs.
`plot_data.csv` and `provenance.json` record all points and selected sources.
The standard `scan_results.csv` is also refreshed from the saved histories.

## Matched Second-Order Trotter Scan And Exact Overlay

`run_qrte_mean_field_active_dt_scan_upto12_trotter2.py` repeats the same eight
cases and all 11 factors (**0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1,
1.25, 1.5, 2**), giving **88 conditions**. Evolution is pinned to symmetric
second-order Trotter with **one slice**. Timesteps, cached Hamiltonians,
saved HF spectra, initial determinants, reference energies, 100-iteration cap,
overlap tolerance `1e-10` and worker settings match the completed exact scan.
The driver checks each condition against the saved exact scan plan; changing
the physical inputs or stopping settings is rejected for this comparison.

The new output root is
`runs/qrte_mean_field_active_dt_scan_upto12_trotter2/factor_<scale>/<case_id>/`.
These are fresh calculations; exact results are never reused as Trotter data.
`scan_plan.csv/.json` and `exact_comparison.json` record the matched conditions
and the exact-plan source hash. Existing exact outputs and plots are preserved.

```bash
# Validate and record all 88 Trotter conditions without propagating or submitting.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_active_dt_scan_upto12_trotter2.py --prepare-only

# Submit four worker jobs, each assigned 22 conditions.
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_dt_scan_upto12_trotter2.sh

# After completion: validate both saved datasets and overlay their log-log curves.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/plot_qrte_mean_field_dt_scan_upto12_exact_vs_trotter2.py
```

Slurm resources remain 36 CPUs, 8G and 48 hours per worker, with the default
array `0-3%4`. The same `QCANT_SCAN_WORKERS` and concurrency controls apply.
Workers resume completed conditions and report failures after trying their
remaining conditions. Configurations pin order 2 and one slice regardless
of inherited evolution/order/slice environment overrides.

The overlay script requires complete, validated results for both scans and
recomputes chemical-accuracy hits from the saved energy histories. It checks
the paired timestep, Hamiltonian/HF hashes, reference energies, active-space
size and stopping settings before plotting. Blue circles show exact QRTE;
orange dashed squares show order-2 Trotter. No-hit conditions use open
triangles and upward arrows in the series color; missing hit iterations stay
blank in the CSVs. No curves are extrapolated for missing runs.

Individual PDF/600-dpi PNG/CSV overlays and `all_systems_loglog.pdf/.png` go to
`runs/qrte_mean_field_active_dt_scan_upto12_exact/exact_vs_trotter2_loglog/`.
`plot_data.csv` contains both series and `provenance.json` records all selected
sources. Standard `scan_results.csv` files are refreshed for both datasets.

## Hexatriene Spin 0/2 Exact And Second-Order Trotter Scans

The hexatriene scans use the same 11 factors **0.01, 0.025, 0.05, 0.1, 0.25,
0.5, 0.75, 1, 1.25, 1.5, 2** times the saved `pi/W_MF`, for two 12-qubit
CASSCF/natural-orbital Hamiltonians with 6 active electrons in 6 orbitals:

| Task indices | Case ID | pi/W_MF (a.u.) | Full timestep range (a.u.) |
| --- | --- | ---: | ---: |
| 0–10 | `hexatriene_equi_cc_pVTZ_0` | 1.009359058538 | 0.010093590585–2.018718117075 |
| 11–21 | `hexatriene_equi_cc_pVTZ_2` | 1.364913481937 | 0.013649134819–2.729826963873 |

Each propagation method has **22 conditions**, for **44 total**. Exact
propagation uses sparse exponential action. Trotter propagation is pinned
to symmetric second order and one slice. Both retain the cached initial
determinants and reference energies, 100-iteration cap, overlap tolerance
`1e-10`, 1.6 mHa stopping threshold, and 18 projection workers. Saved HF
spectra and the validated cache manifest are checked without new SCF/CASSCF.

The separate output roots are `runs/qrte_mean_field_dt_scan_hexatriene_exact`
and `runs/qrte_mean_field_dt_scan_hexatriene_trotter2`, with
`factor_<scale>/<case_id>/` subdirectories. The Trotter driver matches each
condition to the exact plan, including timestep, Hamiltonian/HF hashes,
reference and stopping settings. Existing results are preserved.

```bash
# Prepare both plans, exact first; no molecular calculation or job submission.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_dt_scan_hexatriene_exact.py --prepare-only
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/collect_qrte_mean_field_dt_scan_hexatriene_trotter2.py --prepare-only

# Submit the two independent worker arrays when calculations are requested.
bash scripts/Mushir_script/submit_talon_qrte_mean_field_dt_scan_hexatriene_exact.sh
bash scripts/Mushir_script/submit_talon_qrte_mean_field_dt_scan_hexatriene_trotter2.sh

# After both scans complete: create matched individual and combined log-log overlays.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/plot_qrte_mean_field_dt_scan_hexatriene_exact_vs_trotter2.py
```

Each launcher submits four worker jobs rather than 22 individual jobs;
the worker partitions contain 6, 6, 5, and 5 conditions. Slurm requests
36 CPUs, 8G and 48 hours per worker, and standard `QCANT_SCAN_WORKERS` and
`QCANT_ARRAY_CONCURRENCY` overrides remain available. Completed compatible
conditions are skipped on restart. Plan mismatches require a separate root.

Collectors without `--prepare-only` refresh `scan_results.csv` and plots;
`--loglog` selects log-log axes. The comparison command validates all 44
histories, preserves explicit no-hit bounds, and writes PDF/600-dpi PNG/CSV
files and provenance under
`runs/qrte_mean_field_dt_scan_hexatriene_exact/exact_vs_trotter2_loglog/`.

## Final Exact/Trotter Timestep Comparison Figure

`plot_qrte_dt_scan_publication.py` combines the earlier comparison with
hexatriene spins 0/2 and excludes both BeH2 cases. Its eight panels contain
N2 spins 0/2, hexatriene spins 0/2, H4/STO-3G spin 0, H6/STO-3G spin 0, and
H6/cc-pVTZ spins 0/2. It validates all selected exact/Trotter histories and
paired physical inputs before plotting the 176 selected conditions.

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/plot_qrte_dt_scan_publication.py
```

Outputs share the prefix
`plots/qrte_dt_scan_final/qrte_exact_vs_trotter2_loglog`: vector PDF with
embedded TrueType fonts, editable SVG, 600-dpi PNG, selected plot CSV and
source-provenance JSON. The figure uses shared log-log axes, panel labels,
per-system mean-field timestep annotations and a shared legend. Nested open
triangles and upward arrows show no-hit bounds; hit iterations remain blank
for these conditions. Original plots and calculation outputs are preserved.

## Remaining 17 Systems With The Same Mean-Field Timestep

Prepared driver:
`scripts/Mushir_script/run_qrte_mean_field_active_pi_over_w_remaining17.py`.
It reuses the small10 saved-reference validation, width calculation, and QRTE
runner. Its fixed 17-case array is the complement of small10 within the
validated 27-system cache manifest. It checks that the original 27-case set
is unchanged and retains a stable array order even if manifest rows move.
Ferrocene and `feo4_2minus_ceo` remain excluded.

The saved small10 results were checked before preparation: all ten completed
and reached 1.6 mHa chemical accuracy in 3–10 iterations; their job 11602 logs
contain ten `DONE` records and empty error files.

The remaining-case input audit is saved in
`provenance/qrte_mean_field_active_remaining17_preparation.json` and `.csv`.
At preparation, 15 saved HF spectra passed validation. Magnesium porphyrin
and H8/STO-3G have no `fock_reference.json`: their earlier Fock array tasks
(11280 indices 2 and 23) were cancelled. Their Hamiltonian caches remain valid,
but contain no molecular HF orbital energies. No substitute timestep or new
calculation is used. The full-array submitter fails preflight until both
saved converged spectra are available.

| Array index | Case ID | Saved HF input at preparation |
| --- | --- | --- |
| 0 | `magnesium_porphyrin_ceo` | Missing |
| 1 | `cyclobutadiene_ceo` | Ready |
| 2 | `CH2O_equi_cc_pVTZ_2` | Ready |
| 3 | `CH2O_equi_cc_pVTZ_0` | Ready |
| 4 | `H6_near_equi_cc_pVTZ_2` | Ready |
| 5 | `m_Benzyne_equi_cc_pVTZ_2` | Ready |
| 6 | `hexatriene_equi_cc_pVTZ_2` | Ready |
| 7 | `m_Benzyne_equi_cc_pVTZ_0` | Ready |
| 8 | `FeO` | Ready |
| 9 | `hexatriene_equi_cc_pVTZ_0` | Ready |
| 10 | `H7_near_equi_sto3g_1` | Ready |
| 11 | `fenoco3_minus` | Ready |
| 12 | `feno_2plus` | Ready |
| 13 | `H8_near_equi_sto3g_0` | Missing |
| 14 | `Al3_minus` | Ready |
| 15 | `H9_near_equi_sto3g_1` | Ready |
| 16 | `H10_near_equi_sto3g_0` | Ready |

Read-only input checks report all 17 cases and return a nonzero status if any
saved spectrum is missing or invalid; no output run directories are created:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_pi_over_w_remaining17.py --dry-run

# Check an available case individually, without calculations.
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_qrte_mean_field_active_pi_over_w_remaining17.py \
  --case-id FeO --dry-run
```

Settings remain `dt=pi/W_MF`, symmetric second-order Trotter, one slice,
100 iterations, overlap tolerance 1e-10, and chemical-accuracy stopping at
1.6 mHa against the cached CASSCF/natorb energy. Corrected cyclobutadiene and
m-benzyne indices are preserved. FeO retains its validated ROHF/Bohr provenance;
`feno_2plus` and `fenoco3_minus` retain sfX2C1eROHF. The latter retains its
explicitly recorded unconverged CASSCF cache, while its saved SCF is converged.
ROHF and sfX2C1eROHF use effective Roothaan spatial energies for both spins.
The canonical HF window still need not match optimized CASSCF/AVAS orbitals.

The prepared submission command, for use only when calculations are explicitly
requested and all saved spectra pass preflight, is:

```bash
bash scripts/Mushir_script/submit_talon_qrte_mean_field_active_pi_over_w_remaining17_trotter2.sh
```

It uses a 0–16 array with four concurrent tasks, 36 CPUs, 8G, and 12 hours.
Outputs go to
`runs/qrte_mean_field_active_pi_over_w_remaining17_trotter2/<case_id>/`, with
the same reference/configuration/history/summary files as small10. Logs use
`mushir-qrte-mf-active-rem17-piW-trotter2-*`; its separate job record is
`provenance/latest_qrte_mean_field_active_pi_over_w_remaining17_trotter2_jobid.txt`.
Run and Slurm overrides work as for small10. `QCANT_REFERENCE_ONLY=1` in this
workflow only derives widths from existing saved spectra; it cannot create
the two missing molecular HF spectra.

If explicitly requested later, those two spectra can be generated with the
existing full-Fock reference-only workflow. For the default validated cache
manifest, the two source array indices are 2 and 23:

```bash
# Runs SCF on Talon; submit only after explicit authorization.
QCANT_REFERENCE_ONLY=1 QCANT_OUTDIR=runs/qrte_fock_pi_over_w \
  bash scripts/Mushir_script/submit_talon_qrte_fock_pi_over_w.sh --array=2,23%2
```

Then rerun the 17-case dry-run before submitting QRTE. Preparation itself
does not run SCF, CASSCF, CI, or QRTE and does not submit any jobs.

## N2 PES With A Geometry-Specific Mean-Field Timestep

The new workflow repeats the original MolVQE-style N2 PES, using the exact
Hamiltonian paths recorded in `runs/n2_pes_qrte_natorb_dt0.05`. It preserves
cc-pVTZ, charge 0, 6 active electrons in 6 active orbitals, and spins 0 and 2.
The eight distances are 0.90, 1.00, 1.10, 1.35, 1.50, 1.70, 2.00, and 2.40
Angstrom: 16 array tasks, ordered by increasing distance, spin 0 then spin 2.
The 1.20 Angstrom equilibrium cases remain separate in the existing small10
results, as in the original PES workflow. This is separate from Benchmark-QC.

All 16 original fixed-dt jobs completed; 14 reached chemical accuracy. Spin 2
at 2.00 and 2.40 Angstrom reached the 100-iteration limit without a hit.
Their geometry-specific CASSCF/natural-orbital caches and reference energies
are available and pass input checks. The old caches contain SCF total energies
but no HF orbital-energy spectra, so those spectra must be recovered in jobs.

Each new task runs RHF (spin 0) or ROHF (spin 2) at its own geometry, using
the original SCF settings and checking its total energy against the cached
SCF energy within 1e-6 Eh. It then selects canonical spatial orbitals 4–9
(zero-based), after the four closed-shell core orbitals. At fixed active
occupations `(3,3)` or `(4,2)`, `W_MF` is the highest minus lowest determinant
orbital-energy sum in that six-orbital window, and `dt=pi/W_MF`. ROHF uses
effective Roothaan spatial energies for both spins. This window is an estimate
and need not equal the optimized CASSCF orbital subspace. No equilibrium
spectrum is substituted for another geometry.

QRTE loads the pinned original Hamiltonian cache at that geometry; it does
not rerun CASSCF or diagonalize the CAS. Settings are symmetric second-order
Trotter, one slice, 100 iterations, overlap tolerance 1e-10, and early stopping
at 1.6 mHa relative to the cached CASSCF energy. The original spin-resolved
initial determinant is retained. Generated PES points have no MolVQE dataset
energy; their `error_vs_molvqe_casscf_hartree` CSV column stays blank.

Validate or prepare without running molecular calculations:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_n2_pes_qrte_mean_field_pi_over_w.py --dry-run

/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_n2_pes_qrte_mean_field_pi_over_w.py --prepare-only
```

Preparation writes `pes_plan.json` and `pes_plan.csv` to
`runs/n2_pes_qrte_mean_field_pi_over_w_trotter2/`. The plan pins geometries,
Hamiltonian and original-input hashes, SCF settings, accuracy targets, and
evolution settings. If settings or source files change, use a separate output
directory. Actual timesteps are recorded after each geometry's SCF finishes.

Submit when ready to run calculations:

```bash
bash scripts/Mushir_script/submit_talon_n2_pes_qrte_mean_field_pi_over_w.sh
```

The submitter validates all 16 points and prepares the plan before submitting
array `0-15%4`, with 36 CPUs, 8G, and six hours per task. Extra arguments go to
`sbatch`; common overrides are `QCANT_OUTDIR`, `QCANT_N_STEPS`,
`QCANT_OVERLAP_TOL`, `QCANT_MAX_WORKERS`, `QCANT_ARRAY_CONCURRENCY`, and
`QCANT_SCF_MATCH_TOLERANCE`. `QCANT_REFERENCE_ONLY=1` runs only the molecular
HF step and derives the width/timestep; `QCANT_FORCE=1` recomputes results.

Each `<outdir>/<case_id>/` contains `fock_scf.log`, `fock_reference.json` with
the full HF orbital spectrum, `mean_field_active_reference.json`,
`input_snapshot.json`, `run_config.json`, `min_energy_history.csv`, and
`summary.json`. Matching completed runs and spectra are reused. Failures are
recorded and return a nonzero exit status. Logs use
`mushir-n2-pes-qrte-mf-piW-trotter2-*`; the job record is
`provenance/latest_n2_pes_qrte_mean_field_pi_over_w_jobid.txt`.

The setup tests mock all SCF, QRTE, and Slurm calls:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python -m unittest discover \
  -s scripts/Mushir_script -p test_n2_pes_qrte_mean_field_pi_over_w.py
```

Remake the initial-overlap and fixed-sector-gap iteration plot from saved data:

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/plot_n2_pes_mean_field_overlap_gap_iterations.py
```

This compares geometry-specific `dt=pi/W_MF` QRTE results (order 2,
one slice), fixed-step `dt=0.05` a.u. QRTE (order 1, one slice), and the
original QKUD `epsilon=0.05` results (order 1, one slice).
It includes both 1.20 Angstrom equilibrium cases for all three series, with
the successful 100-step rerun selected for fixed-step equilibrium spin 2.
The two panels plot iteration
counts versus squared initial overlap and `E1-E0` in the fixed active
Nalpha/Nbeta sector, colored by bond distance. All 18 cases and 54 method
points pass history/reference/settings validation. Fixed-step QRTE reaches
chemical accuracy in 16/18 cases; mean-field QRTE and QKUD each reach 18/18.
The fixed-step spin-2 cases at 2.00 and 2.40 Angstrom have no hit within 100
iterations and appear as upward lower-bound arrows, rather than measured hits.
Cache hashes link the saved descriptors to the QRTE Hamiltonians, including
mean-field equilibrium cache copies. The two legacy fixed-step equilibrium
runs lack per-run cache paths/hashes; their shared cache directory and electronic
and geometry input are checked against the descriptor source, and this limited
linkage is recorded in the CSV/JSON. No overlap or gap calculations are rerun.
The figure uses six method/spin markers, a shared legend below the panels,
600-dpi PNG output, and vector PDF output with embedded TrueType fonts.

PNG, PDF, CSV, and source-provenance JSON files share the prefix
`plots/n2_pes_mean_field_pi_over_w_trotter2_vs_qkud_eps0p05_overlap_gap_iterations`.
The original fixed-dt plot is retained.

## CO Spin 2 Using The MolVQE-Matching CASSCF Solution

`run_co_spin2_molvqe_matching.py` uses the second solution from
`diagnose_co_spin2_reference.py`: full-space AVAS with `ncore=0`,
`canonicalize=False`, and `openshell_option=2`, whose orbital ordering is
passed to CASSCF(6e,6o). AVAS itself recommends (8e,7o). The resulting
converged triplet reference is -112.60166830790772 Eh, matching the MolVQE
JSON to about 1.1e-11 Eh. This preparation reproduces the dataset energy;
it does not establish the original authors' precise AVAS settings.

Preparation exports the saved diagnostic checkpoint into the separate cache
`hamiltonian_cache/co_spin2_molvqe_matching/CO_equi_cc_pVTZ_2.npz`.
It validates the complete 225-dimensional fixed-(4 alpha,2 beta) sector,
the triplet spin expectation, checkpoint geometry, and dataset agreement.
The Hamiltonian is explicitly pinned during every propagation call, so the
geometry-based builder cannot substitute the older lower-energy solution.
The original 27-system cache manifest and earlier run directories are preserved.

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_co_spin2_molvqe_matching.py --prepare-only

/home/mushir.kolathingal/miniforge3/envs/expk/bin/python \
  scripts/Mushir_script/run_co_spin2_molvqe_matching.py --dry-run
```

The run plan is `runs/co_spin2_molvqe_matching/run_plan.json`:

| Array task | Method | Parameter | Trotter order | Slices |
| --- | --- | --- | --- | --- |
| 0 | QRTE | dt=0.05 a.u. | 1 | 1 |
| 1 | QRTE | dt=pi/W_MF=0.998530556703 a.u. | 2 | 1 |
| 2 | QKUD | epsilon=0.05 | 1 | 1 |

W_MF=3.14621583936 Eh retains the earlier molecular ROHF canonical-window
estimate: the highest minus lowest orbital-energy sum among determinants
with four alpha and two beta electrons in canonical orbitals 4 through 9
(zero based). It does not reconstruct the optimized AVAS/CASSCF subspace.
Each run stops at 1.6 mHa absolute error against the matching cached
reference, with a default limit of 100 iterations and overlap tolerance 1e-10.

Submit the prepared three-task array when ready:

```bash
bash scripts/Mushir_script/submit_talon_co_spin2_molvqe_matching.sh
```

Each task requests 36 CPUs, 8G, and six hours. Extra submitter arguments go
to `sbatch`. Overrides are `QCANT_OUTDIR`, `QCANT_N_STEPS`,
`QCANT_OVERLAP_TOL`, `QCANT_MAX_WORKERS`, and `QCANT_FORCE`.
The job ID is recorded in `provenance/latest_co_spin2_molvqe_matching_jobid.txt`.
Outputs use separate method directories under `runs/co_spin2_molvqe_matching/`.

```bash
/home/mushir.kolathingal/miniforge3/envs/expk/bin/python -m unittest discover \
  -s scripts/Mushir_script -p test_co_spin2_molvqe_matching.py
```
