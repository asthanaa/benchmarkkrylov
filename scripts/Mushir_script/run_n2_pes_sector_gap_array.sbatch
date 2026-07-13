#!/bin/bash
#SBATCH --job-name=mushir-n2-pes-gap
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
OVERLAP_JSON=runs/n2_pes_initial_state_overlap/${CASE_ID}/initial_overlap.json
OUTPUT=runs/n2_pes_initial_state_gap/${CASE_ID}/sector_gap.json

if [[ -f "${OUTPUT}" ]]; then
  echo "N2 PES sector-gap output already exists for ${CASE_ID}: ${OUTPUT}"
  exit 0
fi

if [[ ! -f "${OVERLAP_JSON}" ]]; then
  echo "Missing N2 PES overlap JSON for ${CASE_ID}: ${OVERLAP_JSON}" >&2
  exit 1
fi

echo "Computing N2 PES sector gap for ${CASE_ID}"
echo "Overlap JSON: ${OVERLAP_JSON}"
echo "Output: ${OUTPUT}"

exec "${PYTHON}" scripts/Mushir_script/compute_sector_gap.py \
  --overlap-json "${OVERLAP_JSON}" \
  --outdir runs/n2_pes_initial_state_gap \
  --nroots 2
