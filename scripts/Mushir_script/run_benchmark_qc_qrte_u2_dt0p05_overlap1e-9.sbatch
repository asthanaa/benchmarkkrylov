#!/bin/bash
#SBATCH --job-name=mushir-bqc-qrte-u2-dt0.05-ov1e-9
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=05:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER:-mushir.kolathingal}}"

mkdir -p logs provenance "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

extra_args=()
if [[ "${BQC_QRTE_U2_FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi
if [[ "${BQC_QRTE_U2_EARLY_STOP:-0}" == "1" ]]; then
  extra_args+=(--early-stop-chemical-accuracy)
fi

exec "${PYTHON}" scripts/Mushir_script/run_benchmark_qc_krylov.py \
  --method qrte \
  --manifest manifests/benchmark_qc_equilibrium_manifest.csv \
  --outdir runs/benchmark_qc_qrte_dt0.05_overlap1e-9 \
  --case-id U2_eq \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --delta-t 0.05 \
  --epsilon 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --overlap-tol 1e-9 \
  --basis-threshold "${QCANT_BASIS_THRESHOLD:-0.0}" \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  "${extra_args[@]}"
