#!/usr/bin/env bash
#SBATCH --job-name=qcant-qkud-pool
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
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
  --method qkud \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir runs/qkud_natorb_eps0.1 \
  --n-steps "${QCANT_N_STEPS:-50}" \
  --epsilon 0.1 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --concurrency "${QCANT_CONCURRENCY:-18}" \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --early-stop-chemical-accuracy \
  --order small-first
