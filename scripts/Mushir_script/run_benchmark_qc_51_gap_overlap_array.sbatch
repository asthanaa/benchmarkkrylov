#!/bin/bash
#SBATCH --job-name=mushir-bqc51-gap-overlap
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-50%6
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-mushir.kolathingal}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"

mkdir -p logs provenance "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
MANIFEST=${BQC51_MANIFEST:-/home/mushir.kolathingal/BenchmarkQC/datasets/benchmarkQC/source/manifest.csv}
BENCHMARK_QC_REPO=${BENCHMARK_QC_REPO:-/home/mushir.kolathingal/BenchmarkQC}
OVERLAP_OUTDIR=${BQC51_OVERLAP_OUTDIR:-runs/benchmark_qc_51_initial_state_overlap}
GAP_OUTDIR=${BQC51_GAP_OUTDIR:-runs/benchmark_qc_51_initial_state_gap}

CASE_ID=$(awk -F, -v task_id="${SLURM_ARRAY_TASK_ID:?}" 'NR == task_id + 2 {print $1}' "${MANIFEST}")
if [[ -z "${CASE_ID}" ]]; then
  printf 'No BenchmarkQC case for SLURM_ARRAY_TASK_ID=%s in %s\n' "${SLURM_ARRAY_TASK_ID}" "${MANIFEST}" >&2
  exit 2
fi

printf 'Computing BenchmarkQC reference-matched gap and initial overlap for %s\n' "${CASE_ID}"
printf 'Overlap output: %s/%s/initial_overlap.json\n' "${OVERLAP_OUTDIR}" "${CASE_ID}"
printf 'Gap output: %s/%s/sector_gap.json\n' "${GAP_OUTDIR}" "${CASE_ID}"

exec "${PYTHON}" scripts/Mushir_script/compute_benchmark_qc_gap_overlap.py \
  --manifest "${MANIFEST}" \
  --benchmark-qc-repo "${BENCHMARK_QC_REPO}" \
  --case-id "${CASE_ID}" \
  --expected-cases 1 \
  --overlap-outdir "${OVERLAP_OUTDIR}" \
  --gap-outdir "${GAP_OUTDIR}" \
  --conv-tol "${BQC51_FCI_CONV_TOL:-1e-12}" \
  --conv-tol-residual "${BQC51_FCI_CONV_TOL_RESIDUAL:-1e-11}" \
  --max-cycle "${BQC51_FCI_MAX_CYCLE:-1000}" \
  --max-space "${BQC51_FCI_MAX_SPACE:-100}" \
  --spin-penalty-shift "${BQC51_FCI_SPIN_PENALTY_SHIFT:-0.5}"
