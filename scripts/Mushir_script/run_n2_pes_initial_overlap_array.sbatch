#!/bin/bash
#SBATCH --job-name=mushir-n2-pes-overlap
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%8
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
  N2_R0p90_cc_pVTZ_0
  N2_R0p90_cc_pVTZ_2
  N2_R1p00_cc_pVTZ_0
  N2_R1p00_cc_pVTZ_2
  N2_R1p10_cc_pVTZ_0
  N2_R1p10_cc_pVTZ_2
  N2_R1p35_cc_pVTZ_0
  N2_R1p35_cc_pVTZ_2
  N2_R1p50_cc_pVTZ_0
  N2_R1p50_cc_pVTZ_2
  N2_R1p70_cc_pVTZ_0
  N2_R1p70_cc_pVTZ_2
  N2_R2p00_cc_pVTZ_0
  N2_R2p00_cc_pVTZ_2
  N2_R2p40_cc_pVTZ_0
  N2_R2p40_cc_pVTZ_2
)

CASE_ID=${CASE_IDS[${SLURM_ARRAY_TASK_ID}]}
RUN_DIR=runs/n2_pes_qkud_natorb_eps0.05/${CASE_ID}
OUTPUT=runs/n2_pes_initial_state_overlap/${CASE_ID}/initial_overlap.json

if [[ -f "${OUTPUT}" ]]; then
  echo "N2 PES overlap output already exists for ${CASE_ID}: ${OUTPUT}"
  exit 0
fi

if [[ ! -f "${RUN_DIR}/summary.json" ]]; then
  echo "Missing N2 PES QKUD summary for ${CASE_ID}: ${RUN_DIR}/summary.json" >&2
  exit 1
fi

echo "Computing N2 PES initial overlap for ${CASE_ID}"
echo "Run directory: ${RUN_DIR}"
echo "Output: ${OUTPUT}"

exec "${PYTHON}" scripts/Mushir_script/compute_initial_overlap.py \
  --run-dir "${RUN_DIR}" \
  --outdir runs/n2_pes_initial_state_overlap \
  --solver pyscf-fci
