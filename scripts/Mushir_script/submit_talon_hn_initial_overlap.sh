#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_hn_initial_overlap_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_hn_initial_overlap_jobid.txt

printf 'H_n initial-overlap array job: %s\n' "${job}"
printf 'Array cases: H4, H5, H6(cc-pVTZ singlet/triplet, sto-3g), H7, H8, H9, H10\n'
printf 'Output: runs/initial_state_overlap/<case_id>/initial_overlap.json\n'
