#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs runs/qrte_natorb_dt0.8

qrte_job=$(sbatch --parsable --export=NIL "${SCRIPT_DIR}/run_qrte_dt0p8_pool.sbatch")
qrte_job=${qrte_job%%;*}

printf 'QRTE delta_t=0.8 job: %s\n' "${qrte_job}"
printf 'QRTE output: runs/qrte_natorb_dt0.8\n'
