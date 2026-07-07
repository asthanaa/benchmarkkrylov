#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_remaining_initial_overlap_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_remaining_initial_overlap_jobid.txt

printf 'Remaining initial-overlap array job: %s\n' "${job}"
printf 'Array size: 17 cases, excluding feo4_2minus_ceo and Ferrocene_ceo\n'
printf 'Output: runs/initial_state_overlap/<case_id>/initial_overlap.json\n'
