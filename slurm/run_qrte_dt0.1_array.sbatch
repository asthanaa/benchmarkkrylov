#!/usr/bin/env bash
#SBATCH --job-name=qcant-qrte-array
#SBATCH --partition=talon-ayush
#SBATCH --nodelist=talon04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=0-28%18
#SBATCH --time=28-00:00:00
#SBATCH --output=/home/ayush.asthana/code/benchmark/logs/%x-%A_%a.out
#SBATCH --error=/home/ayush.asthana/code/benchmark/logs/%x-%A_%a.err

set -euo pipefail

cd /home/ayush.asthana/code/benchmark

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/ayush.asthana/miniconda3/bin/python}

"${PYTHON}" scripts/run_krylov_benchmark.py \
  --method qrte \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir runs/qrte_natorb_dt0.1 \
  --row-index "${SLURM_ARRAY_TASK_ID}" \
  --n-steps 200 \
  --delta-t 0.1 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers 2 \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --early-stop-chemical-accuracy
