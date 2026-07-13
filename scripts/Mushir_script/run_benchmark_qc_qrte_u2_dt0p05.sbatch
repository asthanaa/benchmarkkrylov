#!/bin/bash
#SBATCH --job-name=mushir-bqc-qrte-u2-dt0.05
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

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-${USER:-mushir.kolathingal}}"

mkdir -p logs "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

exec "${PYTHON}" scripts/Mushir_script/run_benchmark_qc_krylov.py \
  --method qrte \
  --manifest manifests/benchmark_qc_equilibrium_manifest.csv \
  --outdir runs/benchmark_qc_qrte_dt0.05 \
  --case-id U2_eq \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --delta-t 0.05 \
  --epsilon 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --basis-threshold "${QCANT_BASIS_THRESHOLD:-0.0}" \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --early-stop-chemical-accuracy \
  --force
