#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_EVOLUTION_MODE=trotter
export QCANT_TROTTER_ORDER=2
export QCANT_DELTA_T=0.05
export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_casscf_natorb_small10_trotter2_dt0p05}"
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_casscf_natorb_pi_over_w_small10.py" \
  --cache-manifest "${CACHE_MANIFEST}" --delta-t "${QCANT_DELTA_T}" --trotter-order 2 --count-cases)
if [[ "${case_count}" != 10 ]]; then
  printf 'Expected ten validated cases, got: %s\n' "${case_count}" >&2
  exit 1
fi

job=$(sbatch --parsable --array="0-9%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_qrte_casscf_natorb_small10_trotter2_dt0p05_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_casscf_natorb_small10_trotter2_dt0p05_jobid.txt
printf 'Second-order QRTE fixed dt=0.05 CASSCF/natorb array job: %s\n' "${job}"
printf 'Validated cases: 10; order: 2; slices per step: %s\n' "${QCANT_TROTTER_STEPS:-1}"
printf 'Output: %s/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s\n' "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
