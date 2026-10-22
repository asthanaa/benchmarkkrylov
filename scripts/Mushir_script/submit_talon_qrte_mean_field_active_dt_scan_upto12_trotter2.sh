#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_mean_field_active_dt_scan_upto12_trotter2}"
export QCANT_SCAN_WORKERS="${QCANT_SCAN_WORKERS:-4}"
if [[ ! "${QCANT_SCAN_WORKERS}" =~ ^[1-9][0-9]*$ ]] || (( QCANT_SCAN_WORKERS > 88 )); then
  printf 'QCANT_SCAN_WORKERS must be an integer between 1 and 88\n' >&2
  exit 1
fi
array_concurrency=${QCANT_ARRAY_CONCURRENCY:-4}
if [[ ! "${array_concurrency}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'QCANT_ARRAY_CONCURRENCY must be a positive integer\n' >&2
  exit 1
fi
if (( array_concurrency > QCANT_SCAN_WORKERS )); then
  array_concurrency=${QCANT_SCAN_WORKERS}
fi
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_mean_field_active_dt_scan_upto12_trotter2.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ "${case_count}" != 88 ]]; then
  printf 'Expected 88 validated scan conditions, got: %s\n' "${case_count}" >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/collect_qrte_mean_field_active_dt_scan_upto12_trotter2.py" --prepare-only \
  --cache-manifest "${CACHE_MANIFEST}" --outdir "${QCANT_OUTDIR}" \
  --n-steps "${QCANT_N_STEPS:-100}" --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --max-workers "${QCANT_MAX_WORKERS:-18}"

job=$(sbatch --parsable --array="0-$((QCANT_SCAN_WORKERS - 1))%${array_concurrency}" \
  "$@" "${SCRIPT_DIR}/run_qrte_mean_field_active_dt_scan_upto12_trotter2_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_mean_field_active_dt_scan_upto12_trotter2_jobid.txt
printf 'Second-order Trotter QRTE timestep scan job: %s\n' "${job}"
printf 'Submitted %s worker jobs; each processes its assigned conditions sequentially\n' "${QCANT_SCAN_WORKERS}"
printf '88 conditions; 8 systems (at most 12 qubits); dt factors: 0.01 0.025 0.05 0.1 0.25 0.5 0.75 1 1.25 1.5 2; second-order Trotter; one slice; matched to exact scan\n'
printf 'Output: %s/factor_<scale>/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s; resources: 8G and 48 hours per task by default\n' \
  "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
