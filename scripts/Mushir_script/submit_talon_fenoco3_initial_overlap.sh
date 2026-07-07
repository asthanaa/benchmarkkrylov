#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_fenoco3_initial_overlap.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_fenoco3_initial_overlap_jobid.txt

printf 'fenoco3_minus initial-overlap job: %s\n' "${job}"
printf 'Output: runs/initial_state_overlap/fenoco3_minus/initial_overlap.json\n'
