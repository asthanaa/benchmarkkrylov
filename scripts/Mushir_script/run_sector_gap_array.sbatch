#!/bin/bash
#SBATCH --job-name=mushir-sector-gap
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-26%4
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
  cyclobutadiene_ceo
  magnesium_porphyrin_ceo
  H5_near_equi_sto3g_1
  N2_equi_cc_pVTZ_2
  CO_equi_cc_pVTZ_2
  N2_equi_cc_pVTZ_0
  CO_equi_cc_pVTZ_0
  BeH2_equi_cc_pVTZ_2
  BeH2_equi_cc_pVTZ_0
  CH2O_equi_cc_pVTZ_2
  CH2O_equi_cc_pVTZ_0
  H6_near_equi_cc_pVTZ_2
  H6_near_equi_cc_pVTZ_0
  H6_near_equi_sto3g_0
  m_Benzyne_equi_cc_pVTZ_2
  m_Benzyne_equi_cc_pVTZ_0
  hexatriene_equi_cc_pVTZ_2
  hexatriene_equi_cc_pVTZ_0
  H7_near_equi_sto3g_1
  H8_near_equi_sto3g_0
  fenoco3_minus
  FeO
  Al3_minus
  feno_2plus
  H9_near_equi_sto3g_1
  H10_near_equi_sto3g_0
)

CASE_ID=${CASE_IDS[${SLURM_ARRAY_TASK_ID}]}
OUTPUT=runs/initial_state_gap/${CASE_ID}/sector_gap.json

if [[ -f "${OUTPUT}" ]]; then
  echo "Sector gap output already exists for ${CASE_ID}: ${OUTPUT}"
  exit 0
fi

echo "Computing sector gap for ${CASE_ID}"
echo "Output: ${OUTPUT}"

exec "${PYTHON}" scripts/Mushir_script/compute_sector_gap.py \
  --case-id "${CASE_ID}" \
  --outdir runs/initial_state_gap \
  --nroots 2
