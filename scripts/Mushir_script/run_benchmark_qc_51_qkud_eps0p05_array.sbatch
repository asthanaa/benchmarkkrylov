#!/bin/bash
#SBATCH --job-name=mushir-bqc51-qkud-eps0.05
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --array=0-50%8
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
MANIFEST=${BQC51_MANIFEST:-/home/mushir.kolathingal/BenchmarkQC/datasets/benchmarkQC/source/manifest.csv}
OUTDIR=${BQC51_QKUD_OUTDIR:-runs/benchmark_qc_51_qkud_eps0.05}

CASE_ID=$(awk -F, -v task_id="${SLURM_ARRAY_TASK_ID:?}" 'NR == task_id + 2 {print $1}' "${MANIFEST}")
if [[ -z "${CASE_ID}" ]]; then
  printf 'No BenchmarkQC case for SLURM_ARRAY_TASK_ID=%s in %s\n' "${SLURM_ARRAY_TASK_ID}" "${MANIFEST}" >&2
  exit 2
fi

exec "${PYTHON}" scripts/Mushir_script/run_benchmark_qc_krylov.py \
  --method qkud \
  --manifest "${MANIFEST}" \
  --case-id "${CASE_ID}" \
  --outdir "${OUTDIR}" \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --delta-t 0.05 \
  --epsilon 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --overlap-tol "${BQC51_OVERLAP_TOL:-1e-10}" \
  --basis-threshold "${QCANT_BASIS_THRESHOLD:-0.0}" \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --early-stop-chemical-accuracy
