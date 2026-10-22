#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${BENCHMARK_ROOT}"
mkdir -p logs provenance
PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}
export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/co_spin2_molvqe_matching}"
"${PYTHON}" "${SCRIPT_DIR}/run_co_spin2_molvqe_matching.py" --prepare-only \
  --outdir "${QCANT_OUTDIR}" --n-steps "${QCANT_N_STEPS:-100}" \
  --overlap-tol "${QCANT_OVERLAP_TOL:-1e-10}" --max-workers "${QCANT_MAX_WORKERS:-18}"
job=$(sbatch --parsable "$@" "${SCRIPT_DIR}/run_co_spin2_molvqe_matching_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_co_spin2_molvqe_matching_jobid.txt
printf 'Submitted CO triplet MolVQE-matching array: %s\n' "${job}"
printf 'Tasks: 0 QRTE dt=0.05/order 1; 1 QRTE pi/W_MF/order 2; 2 QKUD eps=0.05/order 1.\n'
printf 'All tasks use one Trotter slice. Output: %s\n' "${QCANT_OUTDIR}"
