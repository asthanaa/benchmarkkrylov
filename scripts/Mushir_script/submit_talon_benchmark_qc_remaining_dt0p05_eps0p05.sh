#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

qrte_job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_remaining_qrte_dt0p05.sbatch")
qkud_job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_remaining_qkud_eps0p05.sbatch")
qrte_job=${qrte_job%%;*}
qkud_job=${qkud_job%%;*}

printf '%s\n' "${qrte_job}" > provenance/latest_benchmark_qc_remaining_qrte_dt0p05_jobid.txt
printf '%s\n' "${qkud_job}" > provenance/latest_benchmark_qc_remaining_qkud_eps0p05_jobid.txt

printf 'Benchmark-QC remaining QRTE dt=0.05 job: %s\n' "${qrte_job}"
printf 'Benchmark-QC remaining QKUD eps=0.05 job: %s\n' "${qkud_job}"
printf 'Manifest: manifests/benchmark_qc_remaining_geometries_manifest.csv\n'
printf 'QRTE output: runs/benchmark_qc_remaining_qrte_dt0.05\n'
printf 'QKUD output: runs/benchmark_qc_remaining_qkud_eps0.05\n'
printf 'U2 overlap tolerance: 1e-9\n'
