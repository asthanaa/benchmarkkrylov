#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

qrte_job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_qrte_dt0p05.sbatch")
qkud_job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_qkud_eps0p05.sbatch")

printf 'Submitted Benchmark-QC QRTE dt=0.05 job: %s\n' "${qrte_job}"
printf 'Submitted Benchmark-QC QKUD eps=0.05 job: %s\n' "${qkud_job}"
printf 'Output directories:\n'
printf '  runs/benchmark_qc_qrte_dt0.05\n'
printf '  runs/benchmark_qc_qkud_eps0.05\n'
