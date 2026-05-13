#!/usr/bin/env bash
#SBATCH --job-name=qcant-param-pool
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=28-00:00:00
#SBATCH --exclude=talon06,talon07,talon12,talon13,talon14
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

exec "${PYTHON}" scripts/run_param_task_csv_pool.py \
  --task-csv "${QCANT_PARAM_TASK_CSV:-manifests/unconverged_param_probe_0p5_1p0_tasks.csv}" \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --n-steps "${QCANT_N_STEPS:-50}" \
  --concurrency "${QCANT_PARAM_POOL_CONCURRENCY:-1}" \
  --max-workers "${QCANT_PARAM_POOL_MAX_WORKERS:-36}" \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --stop-after-case-hit
