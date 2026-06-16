# QCANT Fig. 1b QRTE/QKUD Benchmark

This repository contains the QRTE/QKUD benchmark workspace for the MolVQE-21
systems from Zenodo record `19051536`. It keeps the source dataset, the manifest
used by the benchmark drivers, the vendored QCANT algorithm code, Slurm launch
scripts, cached Hamiltonians, run outputs, plots, logs, and provenance.

Start with the documentation in `docs/`:

- `docs/repository-map.md` explains what is present where.
- `docs/dataset.md` describes the dataset, manifest, and database layout.
- `docs/algorithms.md` points to the QRTE and QKUD implementations.
- `docs/running.md` explains local and Slurm execution.
- `docs/results.md` explains outputs, logs, plots, caches, and provenance.

The main benchmark entry point is:

```bash
python scripts/run_krylov_benchmark.py --help
```

The Slurm launch point for the full benchmark is:

```bash
cd slurm
./submit_all.sh
```
