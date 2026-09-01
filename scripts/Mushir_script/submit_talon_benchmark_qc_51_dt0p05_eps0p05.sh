#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

qrte_job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_51_qrte_dt0p05_array.sbatch")
qkud_job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_51_qkud_eps0p05_array.sbatch")
qrte_job=${qrte_job%%;*}
qkud_job=${qkud_job%%;*}

printf '%s\n' "${qrte_job}" > provenance/latest_benchmark_qc_51_qrte_dt0p05_jobid.txt
printf '%s\n' "${qkud_job}" > provenance/latest_benchmark_qc_51_qkud_eps0p05_jobid.txt

printf 'BenchmarkQC 51-system QRTE dt=0.05 array job: %s\n' "${qrte_job}"
printf 'BenchmarkQC 51-system QKUD eps=0.05 array job: %s\n' "${qkud_job}"
printf 'Manifest: /home/mushir.kolathingal/BenchmarkQC/datasets/benchmarkQC/source/manifest.csv\n'
printf 'QRTE output: runs/benchmark_qc_51_qrte_dt0.05\n'
printf 'QKUD output: runs/benchmark_qc_51_qkud_eps0.05\n'
