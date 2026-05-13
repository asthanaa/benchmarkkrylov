#!/usr/bin/env bash
#SBATCH --job-name=qcant-adapt-h10
#SBATCH --partition=talon
#SBATCH --exclude=talon04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
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

"${PYTHON}" scripts/run_h10_adapt_vqe.py \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir runs/adapt_h10_qcant \
  --adapt-it 5000 \
  --max-workers 16 \
  --optimizer-maxiter 500 \
  --pool-type fermionic_sd \
  --hamiltonian-source casci
