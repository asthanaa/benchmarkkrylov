# QRTE/QKUD Benchmark Documentation

This repository is a self-contained benchmark workspace for comparing ADAPT
iteration counts against QRTE and QKUD iteration counts on the MolVQE-21 cases
from Zenodo record `19051536`.

The repository is organized around three things:

- A benchmark database under `dataset/zenodo_19051536`.
- A canonical manifest under `manifests/molvqe21_fig1b_manifest.csv`.
- QRTE/QKUD execution code under `scripts/`, `slurm/`, and `vendor/QCANT`.

## Start Here

Use these pages as the main map:

- `repository-map.md` lists every important top-level folder and what belongs there.
- `dataset.md` explains the raw archive, extracted JSON database, and manifest fields.
- `algorithms.md` explains where QRTE and QKUD live and which functions the benchmark calls.
- `running.md` describes local, pooled, and Slurm execution.
- `results.md` explains run directories, summaries, logs, plots, caches, and provenance.

## Core Workflow

The benchmark workflow is:

1. Keep the Zenodo database in `dataset/zenodo_19051536`.
2. Build or inspect the manifest in `manifests/molvqe21_fig1b_manifest.csv`.
3. Run QRTE or QKUD through `scripts/run_krylov_benchmark.py`.
4. Use `scripts/run_krylov_pool.py` or scripts in `slurm/` for larger batches.
5. Collect result tables and plots with `scripts/collect_and_plot.py`.

Completed runs are restartable. If a case already has a completed
`summary.json`, the runner skips it unless `--force` is passed.
