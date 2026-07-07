#!/bin/bash
#SBATCH --job-name=mushir-overlap-hn
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-mushir.kolathingal}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"

mkdir -p logs provenance "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

CASE_IDS=(
  H4_near_equi_sto3g_0
  H5_near_equi_sto3g_1
  H6_near_equi_cc_pVTZ_0
  H6_near_equi_cc_pVTZ_2
  H6_near_equi_sto3g_0
  H7_near_equi_sto3g_1
  H8_near_equi_sto3g_0
  H9_near_equi_sto3g_1
  H10_near_equi_sto3g_0
)

CASE_ID=${CASE_IDS[${SLURM_ARRAY_TASK_ID}]}

exec "${PYTHON}" scripts/Mushir_script/compute_initial_overlap.py \
  --run-dir "runs/qkud_natorb_eps0.05/${CASE_ID}" \
  --outdir runs/initial_state_overlap \
  --solver pyscf-fci
