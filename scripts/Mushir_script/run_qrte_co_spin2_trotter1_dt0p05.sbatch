#!/bin/bash
#SBATCH --job-name=mushir-qrte-co2-dt005-trotter1
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

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

exec "${PYTHON}" scripts/Mushir_script/run_qrte_co_spin2_trotter1_dt0p05.py \
  --cache-manifest "${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}" \
  --outdir "${QCANT_OUTDIR:-runs/qrte_co_spin2_trotter1_dt0p05_correct_reference}" \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --evolution-mode trotter --trotter-order 1 \
  --trotter-steps 1 --delta-t 0.05 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  "${RUN_OPTIONS[@]}"
