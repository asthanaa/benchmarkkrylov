# Repository Map

This page is the high-level map of the repository. Paths are relative to the
repository root.

## Database And Manifest

- `dataset/zenodo_19051536/raw/data_repository.7z` is the raw Zenodo archive.
- `dataset/zenodo_19051536/raw/record_metadata.json` stores metadata for the Zenodo record.
- `dataset/zenodo_19051536/extracted/data_repository` is the extracted benchmark database.
- `dataset/zenodo_19051536/extracted/data_repository/MolVQE-21` contains the JSON systems used by the canonical benchmark manifest.
- `manifests/molvqe21_fig1b_manifest.csv` is the canonical case table consumed by the runners.
- Other files in `manifests/` are task lists and metadata for additional parameter sweeps or follow-up runs.

## Algorithm Code

- `vendor/QCANT/QCANT/qrte` contains the QRTE package.
- `vendor/QCANT/QCANT/qkud` contains the QKUD package.
- `vendor/QCANT/QCANT/qulacs_accel.py` contains the accelerated Qulacs entry points used by this benchmark.
- `vendor/QCANT/QCANT/__init__.py` exports the public QCANT functions used by the drivers.

## Benchmark Drivers

- `scripts/build_manifest.py` builds the canonical MolVQE-21 manifest from the extracted database.
- `scripts/run_krylov_benchmark.py` runs one or more manifest cases for QRTE or QKUD.
- `scripts/run_krylov_pool.py` packs many cases into one allocation and runs several child jobs.
- `scripts/run_krylov_parameter_sweep.py` runs parameter grids for QRTE time steps and QKUD epsilon values.
- `scripts/collect_and_plot.py` reads completed summaries and writes comparison tables and plots.
- `scripts/watch_and_update_plots.py` refreshes plots while jobs are running.
- `scripts/verify_setup.py` performs a lightweight environment check.

## Slurm Launchers

- `slurm/submit_all.sh` submits the main QRTE, QKUD, and plotting jobs.
- `slurm/run_qrte_dt0.1*.sbatch` launch QRTE production jobs with `delta_t=0.1`.
- `slurm/run_qkud_eps0.1*.sbatch` launch QKUD production jobs with `epsilon=0.1`.
- `slurm/run_qrte_delta_sweep_array.sbatch` launches QRTE delta sweeps.
- `slurm/run_qkud_epsilon_sweep_array.sbatch` launches QKUD epsilon sweeps.
- `slurm/make_plots.sbatch` and `slurm/watch_plots_live.sbatch` handle plot generation.

## Outputs And Artifacts

- `runs/` contains per-case snapshots, run configs, histories, summaries, pool logs, and sweep results.
- `logs/` contains Slurm stdout and stderr files.
- `plots/` contains generated CSV, PNG, and PDF outputs.
- `hamiltonian_cache/casscf_natorb` stores cached Hamiltonian arrays for repeated benchmark runs.
- `provenance/` records restarts, source status, patch snapshots, and recovery notes.

## Vendored Package

`vendor/QCANT` is a vendored copy of the QCANT package. The benchmark drivers add
this directory to `sys.path` through the `--qcant-root` option, whose default is
`vendor/QCANT`.
