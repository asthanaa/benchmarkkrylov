#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs runs/qkud_natorb_eps0.5

qkud_job=$(sbatch --parsable --export=NIL "${SCRIPT_DIR}/run_qkud_eps0p5_pool.sbatch")
qkud_job=${qkud_job%%;*}

printf 'QKUD epsilon=0.5 job: %s\n' "${qkud_job}"
printf 'QKUD output: runs/qkud_natorb_eps0.5\n'
