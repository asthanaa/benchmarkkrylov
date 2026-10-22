#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_co_spin2_trotter1_dt0p05_correct_reference}"
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_co_spin2_trotter1_dt0p05.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ "${case_count}" != 1 ]]; then
  printf 'Expected one validated CO case and saved exact spectrum, got: %s\n' "${case_count}" >&2
  exit 1
fi

job=$(sbatch --parsable \
  "$@" "${SCRIPT_DIR}/run_qrte_co_spin2_trotter1_dt0p05.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_co_spin2_trotter1_dt0p05_jobid.txt
printf 'CO spin-2 QRTE dt=0.05 with cached reference, job: %s\n' "${job}"
printf 'Validated cases: 1; cached CASSCF reference; order: 1; slices per step: 1\n'
printf 'Output: %s/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s; chemical accuracy: 1.6 mHa\n' \
  "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
