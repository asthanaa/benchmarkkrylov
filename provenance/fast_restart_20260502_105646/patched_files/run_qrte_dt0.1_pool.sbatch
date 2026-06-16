#!/usr/bin/env bash
#SBATCH --job-name=qcant-qrte-pool
#SBATCH --partition=talon-ayush
#SBATCH --nodelist=talon04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=28-00:00:00
#SBATCH --output=/home/ayush.asthana/code/benchmark/logs/%x-%j.out
#SBATCH --error=/home/ayush.asthana/code/benchmark/logs/%x-%j.err

set -euo pipefail

cd /home/ayush.asthana/code/benchmark

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/ayush.asthana/miniconda3/bin/python}

"${PYTHON}" scripts/run_krylov_pool.py \
  --method qrte \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir runs/qrte_dt0.1 \
  --n-steps 200 \
  --delta-t 0.1 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --concurrency 36 \
  --max-workers 1 \
  --early-stop-chemical-accuracy \
  --order small-first
