#!/bin/bash
#SBATCH --job-name=mushir-bqc-2body-cumulants
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
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
if [[ "${BQC_CUMULANT_FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi

exec "${PYTHON}" scripts/Mushir_script/compute_benchmark_qc_2body_cumulants.py \
  --manifest manifests/benchmark_qc_equilibrium_manifest.csv \
  --outdir runs/benchmark_qc_2body_cumulants \
  --output-csv runs/benchmark_qc_2body_cumulants/two_body_cumulants.csv \
  --case-id N2_eq \
  --case-id FeS_eq \
  --case-id U2_eq \
  --expected-cases 3 \
  --keep-going \
  "${extra_args[@]}"
