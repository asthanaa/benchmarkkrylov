# Algorithms

The benchmark compares two Krylov-style quantum algorithms implemented in the
vendored QCANT package: QRTE and QKUD.

## Where The Code Lives

Core package folders:

- `vendor/QCANT/QCANT/qrte`: QRTE package.
- `vendor/QCANT/QCANT/qkud`: QKUD package.
- `vendor/QCANT/QCANT/qulacs_accel.py`: Qulacs-accelerated implementations used by the benchmark.

Public exports are defined in `vendor/QCANT/QCANT/__init__.py`. The benchmark
uses the Qulacs entry points:

- `QCANT.qrte_qulacs`
- `QCANT.qkud_qulacs`

The non-accelerated package-level implementations are still present in:

- `vendor/QCANT/QCANT/qrte/qrte.py`
- `vendor/QCANT/QCANT/qkud/qkud.py`

## QRTE

QRTE builds a Krylov basis by applying real-time evolution to an initial state.
The benchmark driver calls `QCANT.qrte_qulacs` with:

- molecular symbols and Angstrom geometry from the manifest,
- active electrons and active orbitals from the manifest,
- basis, charge, and spin from the manifest,
- `delta_t` from `--delta-t`,
- `n_steps` from `--n-steps`,
- `evolution_mode` from `--evolution-mode`,
- optional early stopping at chemical accuracy.

The canonical production setting in this workspace is `delta_t=0.1`, reflected
in run folders such as `runs/qrte_dt0.1` and Slurm files such as
`slurm/run_qrte_dt0.1_pool.sbatch`.

## QKUD

QKUD builds a Krylov basis using a unitary-decomposition recurrence. The
benchmark driver calls `QCANT.qkud_qulacs` with the same molecular and
active-space data as QRTE, but uses:

- `epsilon` from `--epsilon`,
- `n_steps` from `--n-steps`,
- the same cache and early-stopping options.

The canonical production setting in this workspace is `epsilon=0.1`, reflected
in run folders such as `runs/qkud_eps0.1` and Slurm files such as
`slurm/run_qkud_eps0.1_pool.sbatch`.

## Driver Integration

`scripts/run_krylov_benchmark.py` is the main interface between the manifest and
QCANT. It:

1. Reads `manifests/molvqe21_fig1b_manifest.csv`.
2. Selects cases by row index, case id, or limit.
3. Imports QCANT from `vendor/QCANT` unless `--qcant-root` overrides it.
4. Calls `QCANT.qrte_qulacs` for `--method qrte`.
5. Calls `QCANT.qkud_qulacs` for `--method qkud`.
6. Writes input snapshots, run configs, minimum-energy histories, and summaries.

The driver prefers cached Hamiltonians from:

```text
hamiltonian_cache/casscf_natorb
```

The default Hamiltonian source is `casscf_natorb`, and the active-orbital labels
for selected cases are embedded in the runner.

## What Counts As A Hit

For each method, a chemical-accuracy hit is the first iteration where the
minimum projected energy is within `0.0016 Eh` of the CASSCF reference energy.
The driver writes this value as `hit_iteration` in each `summary.json`.
