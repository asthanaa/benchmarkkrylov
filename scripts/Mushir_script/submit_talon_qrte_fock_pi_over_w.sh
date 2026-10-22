#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_fock_pi_over_w.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ ! "${case_count}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Invalid validated case count: %s\n' "${case_count}" >&2
  exit 1
fi

job=$(sbatch --parsable --array="0-$((case_count - 1))%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_qrte_fock_pi_over_w_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_fock_pi_over_w_jobid.txt
printf 'QRTE dt=pi/W Fock array job: %s\n' "${job}"
printf 'Validated cases: %s; W=max(full SCF orbital energy)-min(full SCF orbital energy)\n' "${case_count}"
printf 'Output: %s/<case_id>/\n' "${QCANT_OUTDIR:-runs/qrte_fock_pi_over_w}"
printf 'Maximum iterations: %s; overlap tolerance: %s\n' "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
