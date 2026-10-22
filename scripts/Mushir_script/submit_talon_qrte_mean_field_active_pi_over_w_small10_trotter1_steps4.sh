#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_mean_field_active_pi_over_w_small10_trotter1_steps4}"
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
CACHE_MANIFEST=${QCANT_CACHE_MANIFEST:-hamiltonian_cache/correct_27_systems/manifest.csv}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4.py" \
  --cache-manifest "${CACHE_MANIFEST}" --count-cases)
if [[ "${case_count}" != 10 ]]; then
  printf 'Expected ten validated cases and saved mean-field references, got: %s\n' "${case_count}" >&2
  exit 1
fi

job=$(sbatch --parsable --array="0-9%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_mean_field_active_pi_over_w_small10_trotter1_steps4_jobid.txt
printf 'First-order QRTE dt=pi/W_MF active-window array job: %s\n' "${job}"
printf 'Validated cases: 10; saved RHF/ROHF spectra; order: 1; slices per step: %s\n' "4"
printf 'Output: %s/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s; chemical accuracy: 1.6 mHa\n' \
  "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
