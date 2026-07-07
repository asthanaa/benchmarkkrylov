#!/bin/bash
#SBATCH --job-name=mushir-overlap-rest
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-16%4
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
  cyclobutadiene_ceo
  magnesium_porphyrin_ceo
  N2_equi_cc_pVTZ_2
  CO_equi_cc_pVTZ_2
  N2_equi_cc_pVTZ_0
  CO_equi_cc_pVTZ_0
  BeH2_equi_cc_pVTZ_2
  BeH2_equi_cc_pVTZ_0
  CH2O_equi_cc_pVTZ_2
  CH2O_equi_cc_pVTZ_0
  m_Benzyne_equi_cc_pVTZ_2
  m_Benzyne_equi_cc_pVTZ_0
  hexatriene_equi_cc_pVTZ_2
  hexatriene_equi_cc_pVTZ_0
  FeO
  Al3_minus
  feno_2plus
)

RUN_DIRS=(
  runs/qkud_corrected_active_space_eps0.05/cyclobutadiene_ceo
  runs/qkud_natorb_eps0.05/magnesium_porphyrin_ceo
  runs/qkud_natorb_eps0.05/N2_equi_cc_pVTZ_2
  runs/qkud_natorb_eps0.05/CO_equi_cc_pVTZ_2
  runs/qkud_natorb_eps0.05/N2_equi_cc_pVTZ_0
  runs/qkud_natorb_eps0.05/CO_equi_cc_pVTZ_0
  runs/qkud_natorb_eps0.05/BeH2_equi_cc_pVTZ_2
  runs/qkud_natorb_eps0.05/BeH2_equi_cc_pVTZ_0
  runs/qkud_natorb_eps0.05/CH2O_equi_cc_pVTZ_2
  runs/qkud_natorb_eps0.05/CH2O_equi_cc_pVTZ_0
  runs/qkud_corrected_active_space_eps0.05/m_Benzyne_equi_cc_pVTZ_2
  runs/qkud_corrected_active_space_eps0.05/m_Benzyne_equi_cc_pVTZ_0
  runs/qkud_natorb_eps0.05/hexatriene_equi_cc_pVTZ_2
  runs/qkud_natorb_eps0.05/hexatriene_equi_cc_pVTZ_0
  runs/qkud_fe_eps0.05/FeO
  runs/qkud_natorb_eps0.05/Al3_minus
  runs/qkud_fe_eps0.05/feno_2plus
)

CASE_ID=${CASE_IDS[${SLURM_ARRAY_TASK_ID}]}
RUN_DIR=${RUN_DIRS[${SLURM_ARRAY_TASK_ID}]}
OUTPUT=runs/initial_state_overlap/${CASE_ID}/initial_overlap.json

if [[ -f "${OUTPUT}" ]]; then
  echo "Overlap output already exists for ${CASE_ID}: ${OUTPUT}"
  exit 0
fi

echo "Computing initial overlap for ${CASE_ID}"
echo "Run directory: ${RUN_DIR}"
echo "Output: ${OUTPUT}"

exec "${PYTHON}" scripts/Mushir_script/compute_initial_overlap.py \
  --run-dir "${RUN_DIR}" \
  --outdir runs/initial_state_overlap \
  --solver pyscf-fci
