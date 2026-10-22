#!/bin/bash
#SBATCH --job-name=mushir-co2-molvqe-matching
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail
cd /home/mushir.kolathingal/benchmarkkrylov
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"
mkdir -p logs provenance "${PYTHONPYCACHEPREFIX}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
RUN_OPTIONS=()
if [[ "${QCANT_FORCE:-0}" == 1 ]]; then
  RUN_OPTIONS+=(--force)
fi
exec "${PYTHON}" scripts/Mushir_script/run_co_spin2_molvqe_matching.py \
  --task-index "${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}" \
  --outdir "${QCANT_OUTDIR:-runs/co_spin2_molvqe_matching}" \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  "${RUN_OPTIONS[@]}"
