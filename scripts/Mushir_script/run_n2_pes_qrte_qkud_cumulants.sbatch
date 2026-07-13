#!/bin/bash
#SBATCH --job-name=mushir-n2-pes-dt0.05-eps0.05
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
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

exec "${PYTHON}" scripts/Mushir_script/run_n2_pes_qrte_qkud_cumulants.py \
  --mode "${N2_PES_MODE:-all}" \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --delta-t "${QCANT_DELTA_T:-0.05}" \
  --epsilon "${QCANT_EPSILON:-0.05}" \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --early-stop-chemical-accuracy \
  --keep-going
