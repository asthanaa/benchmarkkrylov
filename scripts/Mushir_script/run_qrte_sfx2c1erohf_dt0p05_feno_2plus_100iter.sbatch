#!/bin/bash
#SBATCH --job-name=mushir-qrte-x2c-feno-dt0.05-100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=12:00:00
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

exec "${PYTHON}" scripts/Mushir_script/run_qrte_sfx2c1erohf_dt0p05_cases.py \
  --case-id feno_2plus \
  --outdir runs/qrte_sfx2c1erohf_natorb_dt0.05_100iter \
  --n-steps 100 \
  --delta-t 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb_sfx2c1erohf
