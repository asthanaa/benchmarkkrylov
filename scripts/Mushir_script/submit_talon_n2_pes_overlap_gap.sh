#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

overlap_job=$(sbatch --parsable "${SCRIPT_DIR}/run_n2_pes_initial_overlap_array.sbatch")
overlap_job=${overlap_job%%;*}
printf '%s\n' "${overlap_job}" > provenance/latest_n2_pes_initial_overlap_jobid.txt

gap_job=$(sbatch --parsable --dependency=afterok:${overlap_job} "${SCRIPT_DIR}/run_n2_pes_sector_gap_array.sbatch")
gap_job=${gap_job%%;*}
printf '%s\n' "${gap_job}" > provenance/latest_n2_pes_sector_gap_jobid.txt
printf 'overlap=%s\ngap=%s\n' "${overlap_job}" "${gap_job}" > provenance/latest_n2_pes_overlap_gap_jobids.txt

printf 'N2 PES initial-overlap array job: %s\n' "${overlap_job}"
printf 'N2 PES sector-gap array job: %s (afterok:%s)\n' "${gap_job}" "${overlap_job}"
printf 'Array size: 16 cases, distances 0.90-2.40 A and spins 0/2\n'
printf 'Overlap output: runs/n2_pes_initial_state_overlap/<case_id>/initial_overlap.json\n'
printf 'Gap output: runs/n2_pes_initial_state_gap/<case_id>/sector_gap.json\n'
