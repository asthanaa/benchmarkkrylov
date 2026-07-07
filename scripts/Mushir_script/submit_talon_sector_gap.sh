#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_sector_gap_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_sector_gap_jobid.txt

printf 'Sector-gap array job: %s\n' "${job}"
printf 'Array size: 27 cases, excluding feo4_2minus_ceo and Ferrocene_ceo\n'
printf 'Output: runs/initial_state_gap/<case_id>/sector_gap.json\n'
