#!/usr/bin/env bash
#SBATCH --job-name=qcant-qrte-dt-sweep
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --array=0-3
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
SWEEP_VALUES=${QCANT_SWEEP_VALUES:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0}
SHARD_COUNT=${QCANT_SWEEP_SHARDS:-4}

"${PYTHON}" scripts/run_krylov_parameter_sweep.py \
  --method qrte \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --baseline-dir runs/qrte_natorb_dt0.1 \
  --outdir runs/qrte_natorb_delta_sweep \
  --values "${SWEEP_VALUES}" \
  --n-steps 50 \
  --delta-t 0.1 \
  --epsilon 0.1 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --concurrency 36 \
  --max-workers 1 \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --shard-count "${SHARD_COUNT}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
