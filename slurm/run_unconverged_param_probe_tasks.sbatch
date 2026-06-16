#!/usr/bin/env bash
#SBATCH --job-name=qcant-param-task
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=28-00:00:00
#SBATCH --array=0-35%8
#SBATCH --exclude=talon06,talon07,talon12,talon13,talon14
#SBATCH --output=/home/ayush.asthana/code/benchmark/logs/%x-%A_%a.out
#SBATCH --error=/home/ayush.asthana/code/benchmark/logs/%x-%A_%a.err

set -euo pipefail

cd /home/ayush.asthana/code/benchmark

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/ayush.asthana/miniconda3/bin/python}
TASK_CSV=${QCANT_PARAM_TASK_CSV:-manifests/unconverged_param_probe_0p5_1p0_tasks.csv}

line=$(awk -F, -v idx="$SLURM_ARRAY_TASK_ID" 'NR == idx + 2 {print}' "$TASK_CSV")
if [[ -z "${line}" ]]; then
  echo "No task for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

IFS=, read -r METHOD CASE_ID PARAMETER_NAME PARAMETER_VALUE OUTDIR SIZE_KEY <<<"${line}"

DELTA_T=0.1
EPSILON=0.1
case "${PARAMETER_NAME}" in
  delta_t) DELTA_T="${PARAMETER_VALUE}" ;;
  epsilon) EPSILON="${PARAMETER_VALUE}" ;;
  *)
    echo "Unexpected parameter ${PARAMETER_NAME}" >&2
    exit 2
    ;;
esac

echo "TASK index=${SLURM_ARRAY_TASK_ID} method=${METHOD} case=${CASE_ID} ${PARAMETER_NAME}=${PARAMETER_VALUE} size=${SIZE_KEY}"

exec "${PYTHON}" scripts/run_krylov_benchmark.py \
  --single-case \
  --method "${METHOD}" \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir "${OUTDIR}" \
  --case-id "${CASE_ID}" \
  --n-steps "${QCANT_N_STEPS:-50}" \
  --delta-t "${DELTA_T}" \
  --epsilon "${EPSILON}" \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_PARAM_MAX_WORKERS:-36}" \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --early-stop-chemical-accuracy
