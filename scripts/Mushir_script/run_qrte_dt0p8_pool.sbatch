#!/bin/bash
#SBATCH --job-name=mushir-qrte-dt0.8
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

exec "${PYTHON}" scripts/run_krylov_pool.py \
  --method qrte \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir runs/qrte_natorb_dt0.8 \
  --n-steps "${QCANT_N_STEPS:-50}" \
  --delta-t 0.8 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --concurrency "${QCANT_CONCURRENCY:-18}" \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --early-stop-chemical-accuracy \
  --order small-first
