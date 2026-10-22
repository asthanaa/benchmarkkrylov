#!/bin/bash
#SBATCH --job-name=mushir-qrte-fock-piW
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-26%4
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail
cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-mushir.kolathingal}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"
mkdir -p logs provenance "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
RUN_OPTIONS=()
if [[ "${QCANT_REFERENCE_ONLY:-0}" == 1 ]]; then
  RUN_OPTIONS+=(--reference-only)
fi
if [[ "${QCANT_FORCE:-0}" == 1 ]]; then
  RUN_OPTIONS+=(--force)
fi

exec "${PYTHON}" scripts/Mushir_script/run_qrte_fock_pi_over_w.py \
  --case-index "${SLURM_ARRAY_TASK_ID:?Slurm array task ID is required}" \
  --cache-manifest "${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}" \
  --outdir "${QCANT_OUTDIR:-runs/qrte_fock_pi_over_w}" \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --evolution-mode "${QCANT_EVOLUTION_MODE:-trotter}" \
  --trotter-steps "${QCANT_TROTTER_STEPS:-1}" \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --scf-match-tolerance "${QCANT_SCF_MATCH_TOLERANCE:-1e-6}" \
  "${RUN_OPTIONS[@]}"
