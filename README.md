# QCANT Fig. 1b QRTE/QKUD Benchmark

This directory stages the Zenodo record `19051536`, vendors QCANT, and runs a
restartable MolVQE-21 benchmark comparing ADAPT iteration counts with QRTE and
QKUD iteration counts to chemical accuracy.

## Data and Manifest

- Raw archive: `dataset/zenodo_19051536/raw/data_repository.7z`
- Extracted dataset: `dataset/zenodo_19051536/extracted/data_repository`
- Canonical manifest: `manifests/molvqe21_fig1b_manifest.csv`

The manifest uses the top-level `MolVQE-21` JSON files, computes `h0.25` from
the CASSCF CI vector, and records the first stored ADAPT iteration reaching
`1.6e-3 Eh` relative to the CASSCF energy.

## Run

Submit the full QRTE, QKUD, and dependent plotting jobs:

```bash
cd /home/ayush.asthana/code/benchmark/slurm
./submit_all.sh
```

The Slurm jobs use the `talon-ayush` partition on `talon04` as QRTE/QKUD job
arrays, with one manifest row per array task. Each task requests 2 CPUs and
writes restart files under `runs/qrte_dt0.1` and `runs/qkud_eps0.1`.

## Resume

Rerun the same Slurm scripts or local driver commands. Cases with a completed
`summary.json` are skipped automatically.

## Plot Outputs

- `plots/adapt_vs_qrte_iterations_vs_entropy.csv`
- `plots/adapt_vs_qrte_iterations_vs_entropy.png`
- `plots/adapt_vs_qrte_iterations_vs_entropy.pdf`
- `plots/adapt_vs_qkud_iterations_vs_entropy.csv`
- `plots/adapt_vs_qkud_iterations_vs_entropy.png`
- `plots/adapt_vs_qkud_iterations_vs_entropy.pdf`
