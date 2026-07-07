#!/bin/bash
#SBATCH --job-name=mushir-qrte-x2c-feno-dtgrid-100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/home/mushir.kolathingal/miniforge3/envs/expk/bin/python

DT_VALUES=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
DT_LABELS=(dt0p01 dt0p02 dt0p03 dt0p04 dt0p05 dt0p06 dt0p07 dt0p08 dt0p09 dt0p10)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DELTA_T=${DT_VALUES[${TASK_ID}]}
DT_LABEL=${DT_LABELS[${TASK_ID}]}

exec "${PYTHON}" scripts/Mushir_script/run_qrte_sfx2c1erohf_dt0p05_cases.py \
  --case-id feno_2plus \
  --outdir "runs/qrte_sfx2c1erohf_natorb_feno_dt0.01_to_0.10_100iter/${DT_LABEL}" \
  --n-steps 100 \
  --delta-t "${DELTA_T}" \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb_sfx2c1erohf
