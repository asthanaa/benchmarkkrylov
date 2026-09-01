#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_51_gap_overlap_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_medora_benchmark_qc_51_gap_overlap_jobid.txt

printf 'Medora BenchmarkQC reference-matched gap/overlap array job: %s\n' "${job}"
printf 'Array size: 51 cases\n'
printf 'Overlap output: runs/benchmark_qc_51_initial_state_overlap/<case_id>/initial_overlap.json\n'
printf 'Gap output: runs/benchmark_qc_51_initial_state_gap/<case_id>/sector_gap.json\n'
