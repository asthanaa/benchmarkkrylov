#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_uhf_determinant_pi_over_w_small10.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ "${case_count}" != 10 ]]; then
  printf 'Expected ten validated cases, got: %s\n' "${case_count}" >&2
  exit 1
fi

job=$(sbatch --parsable --array="0-9%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_qrte_uhf_determinant_pi_over_w_small10_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_uhf_determinant_pi_over_w_small10_jobid.txt
printf 'QRTE dt=pi/W UHF determinant array job: %s\n' "${job}"
printf 'Validated cases: 10; W=highest-minus-lowest full-system frozen UHF determinant energy\n'
printf 'Output: %s/<case_id>/\n' "${QCANT_OUTDIR:-runs/qrte_uhf_determinant_pi_over_w_small10}"
printf 'Maximum iterations: %s; overlap tolerance: %s\n' "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
