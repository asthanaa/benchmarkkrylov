#!/bin/bash
#SBATCH --job-name=mushir-bqc-rem-qrte-dt0.05
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=2-00:00:00
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
MANIFEST=manifests/benchmark_qc_remaining_geometries_manifest.csv
OUTDIR=runs/benchmark_qc_remaining_qrte_dt0.05

COMMON_ARGS=(
  --method qrte
  --manifest "${MANIFEST}"
  --outdir "${OUTDIR}"
  --n-steps "${QCANT_N_STEPS:-100}"
  --delta-t 0.05
  --epsilon 0.05
  --evolution-mode trotter
  --trotter-steps 1
  --basis-threshold "${QCANT_BASIS_THRESHOLD:-0.0}"
  --max-workers "${QCANT_MAX_WORKERS:-2}"
)

if [[ "${BQC_REMAINING_FORCE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--force)
fi
if [[ "${BQC_REMAINING_EARLY_STOP:-1}" == "1" ]]; then
  COMMON_ARGS+=(--early-stop-chemical-accuracy)
fi

NON_U2_CASE_IDS=(
  N2_b0p8
  N2_b1p0
  N2_b1p2
  N2_b1p6
  N2_b1p8
  N2_b2p0
  N2_b2p2
  N2_b2p4
  N2_b2p6
  N2_b2p8
  FeS_b1p826
  FeS_b2p0
  FeS_b2p2
  FeS_b2p6
  FeS_b2p8
  FeS_b3p0
  FeS_b3p2
  FeS_b3p6
  FeS_b3p7
  FeS_b4p0
  FeS_b4p2
  FeS_b4p5
  FeS_b4p8
)

U2_CASE_IDS=(
  U2_b2p0_idx0
  U2_b2p2
  U2_b2p4
  U2_b2p6
  U2_b2p8
  U2_b2p0_idx6
)

status=0

case_args=()
for case_id in "${NON_U2_CASE_IDS[@]}"; do
  case_args+=(--case-id "${case_id}")
done
"${PYTHON}" scripts/Mushir_script/run_benchmark_qc_krylov.py \
  "${COMMON_ARGS[@]}" \
  --overlap-tol "${BQC_NON_U2_OVERLAP_TOL:-1e-10}" \
  "${case_args[@]}" || status=$?

case_args=()
for case_id in "${U2_CASE_IDS[@]}"; do
  case_args+=(--case-id "${case_id}")
done
"${PYTHON}" scripts/Mushir_script/run_benchmark_qc_krylov.py \
  "${COMMON_ARGS[@]}" \
  --overlap-tol "${BQC_U2_OVERLAP_TOL:-1e-9}" \
  "${case_args[@]}" || status=$?

exit "${status}"
