# Running The Benchmark

The benchmark can run one case locally, a small batch locally, or large batches
through Slurm.

## Environment Expectations

The runner imports QCANT from `vendor/QCANT` by default and requires the Qulacs
runtime stack. The runner sets common BLAS/OpenMP thread environment variables
to `1` unless they are already set.

Use this lightweight check first:

```bash
python scripts/verify_setup.py
```

## Local Single-Method Runs

Show available options:

```bash
python scripts/run_krylov_benchmark.py --help
```

Run the first manifest row with QRTE:

```bash
python scripts/run_krylov_benchmark.py \
  --method qrte \
  --outdir runs/qrte_dt0.1 \
  --row-index 0 \
  --delta-t 0.1 \
  --n-steps 200 \
  --early-stop-chemical-accuracy
```

Run the first manifest row with QKUD:

```bash
python scripts/run_krylov_benchmark.py \
  --method qkud \
  --outdir runs/qkud_eps0.1 \
  --row-index 0 \
  --epsilon 0.1 \
  --n-steps 200 \
  --early-stop-chemical-accuracy
```

Useful selectors:

- `--row-index N`: run one zero-based manifest row.
- `--case-id CASE`: run a named case; repeat the flag for multiple cases.
- `--limit N`: run the first `N` selected rows.
- `--force`: rerun even when `summary.json` is already completed.

## Pooled Runs

`scripts/run_krylov_pool.py` runs multiple cases inside one allocation. It
launches child `run_krylov_benchmark.py` processes and writes pool progress into
the output directory.

Example:

```bash
python scripts/run_krylov_pool.py \
  --method qrte \
  --outdir runs/qrte_dt0.1 \
  --delta-t 0.1 \
  --n-steps 200 \
  --concurrency 4 \
  --max-workers 1 \
  --early-stop-chemical-accuracy
```

Important pool options:

- `--concurrency`: number of molecules run at the same time.
- `--max-workers`: QCANT workers per molecule.
- `--order small-first`: default scheduling order for faster early progress.
- `--pool-id`: identifier used in pool progress filenames.

## Slurm Runs

The full benchmark launch point is:

```bash
cd slurm
./submit_all.sh
```

Production QRTE scripts use `delta_t=0.1`; production QKUD scripts use
`epsilon=0.1`. Parameter sweep scripts explore additional values.

Common Slurm files:

- `slurm/run_qrte_dt0.1_pool.sbatch`
- `slurm/run_qkud_eps0.1_pool.sbatch`
- `slurm/run_qrte_delta_sweep_array.sbatch`
- `slurm/run_qkud_epsilon_sweep_array.sbatch`
- `slurm/make_plots.sbatch`
- `slurm/watch_plots_live.sbatch`

## Resume Behavior

Each case has a `summary.json`. If that file exists and has
`"status": "completed"`, the drivers skip the case. This makes restarts simple:
rerun the same command or Slurm script and only missing or failed cases continue.

Use `--force` only when intentionally replacing existing outputs.

## Plot Refresh

After runs complete, regenerate tables and figures with:

```bash
python scripts/collect_and_plot.py
```

For live updates during long jobs, use:

```bash
python scripts/watch_and_update_plots.py
```
