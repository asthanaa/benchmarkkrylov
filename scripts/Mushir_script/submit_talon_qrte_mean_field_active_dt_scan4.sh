#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_mean_field_active_dt_scan4_trotter2}"
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_mean_field_active_dt_scan4.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ "${case_count}" != 28 ]]; then
  printf 'Expected 28 validated scan conditions, got: %s\n' "${case_count}" >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/collect_qrte_mean_field_active_dt_scan4.py" --prepare-only \
  --cache-manifest "${CACHE_MANIFEST}" --outdir "${QCANT_OUTDIR}" \
  --n-steps "${QCANT_N_STEPS:-100}" --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --max-workers "${QCANT_MAX_WORKERS:-18}"

job=$(sbatch --parsable --array="0-27%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_qrte_mean_field_active_dt_scan4_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_mean_field_active_dt_scan4_jobid.txt
printf 'QRTE timestep scan job: %s\n' "${job}"
printf '28 conditions; 4 systems; dt factors: 0.25 0.5 0.75 1 1.25 1.5 2; order 2; one slice\n'
printf 'Output: %s/factor_<scale>/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s; time limit: 6 hours by default\n' \
  "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
