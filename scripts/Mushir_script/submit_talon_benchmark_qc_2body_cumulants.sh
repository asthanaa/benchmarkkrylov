#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_2body_cumulants.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_benchmark_qc_2body_cumulants_jobid.txt

printf 'Benchmark-QC two-body cumulant job: %s\n' "${job}"
printf 'Cases: N2_eq, FeS_eq, U2_eq\n'
printf 'Output: runs/benchmark_qc_2body_cumulants/<case_id>/cumulant_2body.json\n'
printf 'CSV: runs/benchmark_qc_2body_cumulants/two_body_cumulants.csv\n'
