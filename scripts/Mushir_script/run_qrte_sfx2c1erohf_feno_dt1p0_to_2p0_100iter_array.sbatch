#!/bin/bash
#SBATCH --job-name=mushir-qrte-x2c-feno-dt1-2-100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-${USER:-mushir.kolathingal}
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

DT_VALUES=(1.00 1.25 1.50 1.75 2.00)
DT_LABELS=(dt1p0 dt1p25 dt1p5 dt1p75 dt2p0)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DELTA_T=${DT_VALUES[${TASK_ID}]}
DT_LABEL=${DT_LABELS[${TASK_ID}]}

printf 'Running FeNO2+ sfX2C1eROHF QRTE 100-iteration task %s: %s\n' "${TASK_ID}" "${DT_LABEL}"

exec "${PYTHON}" scripts/Mushir_script/run_qrte_sfx2c1erohf_dt0p05_cases.py \
  --case-id feno_2plus \
  --outdir "runs/qrte_sfx2c1erohf_natorb_feno_dt1.0_to_2.0_100iter/${DT_LABEL}" \
  --n-steps 100 \
  --delta-t "${DELTA_T}" \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb_sfx2c1erohf
