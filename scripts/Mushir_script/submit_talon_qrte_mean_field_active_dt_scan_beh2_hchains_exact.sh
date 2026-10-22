#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_mean_field_active_dt_scan_beh2_hchains_exact}"
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_mean_field_active_dt_scan_beh2_hchains_exact.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ "${case_count}" != 49 ]]; then
  printf 'Expected 49 validated scan conditions, got: %s\n' "${case_count}" >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/collect_qrte_mean_field_active_dt_scan_beh2_hchains_exact.py" --prepare-only \
  --cache-manifest "${CACHE_MANIFEST}" --outdir "${QCANT_OUTDIR}" \
  --n-steps "${QCANT_N_STEPS:-100}" --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --max-workers "${QCANT_MAX_WORKERS:-18}"

job=$(sbatch --parsable --array="0-48%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_qrte_mean_field_active_dt_scan_beh2_hchains_exact_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_mean_field_active_dt_scan_beh2_hchains_exact_jobid.txt
printf 'Exact exponential QRTE timestep scan job: %s\n' "${job}"
printf '49 conditions; 7 systems; dt factors: 0.25 0.5 0.75 1 1.25 1.5 2; exact exponential; sparse expm_multiply; no Trotterization\n'
printf 'Output: %s/factor_<scale>/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s; resources: 8G and 48 hours per task by default\n' \
  "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
