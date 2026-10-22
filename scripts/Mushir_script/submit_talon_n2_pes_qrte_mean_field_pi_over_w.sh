#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance

export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/n2_pes_qrte_mean_field_pi_over_w_trotter2}"
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
case_count=$("${PYTHON}" "${SCRIPT_DIR}/run_n2_pes_qrte_mean_field_pi_over_w.py" --count-cases)
if [[ "${case_count}" != 16 ]]; then
  printf 'Expected 16 validated PES points, got: %s\n' "${case_count}" >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/run_n2_pes_qrte_mean_field_pi_over_w.py" --prepare-only \
  --outdir "${QCANT_OUTDIR}" --n-steps "${QCANT_N_STEPS:-100}" \
  --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --scf-match-tolerance "${QCANT_SCF_MATCH_TOLERANCE:-1e-6}"

job=$(sbatch --parsable --array="0-15%${QCANT_ARRAY_CONCURRENCY:-4}" \
  "$@" "${SCRIPT_DIR}/run_n2_pes_qrte_mean_field_pi_over_w_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_n2_pes_qrte_mean_field_pi_over_w_jobid.txt
printf 'N2 PES QRTE job: %s\n' "${job}"
printf '16 points; RHF/ROHF at each geometry; dt=pi/W_MF; order 2; one slice\n'
printf 'Output: %s/<case_id>/\n' "${QCANT_OUTDIR}"
printf 'Maximum iterations: %s; overlap tolerance: %s; time limit: 6 hours by default\n' \
  "${QCANT_N_STEPS:-100}" "${QCANT_OVERLAP_TOL:-1e-10}"
