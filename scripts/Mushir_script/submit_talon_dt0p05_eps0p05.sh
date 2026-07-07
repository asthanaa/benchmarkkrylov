#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

qrte_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_dt0p05_pool.sbatch")
qkud_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qkud_eps0p05_pool.sbatch")
qrte_job=${qrte_job%%;*}
qkud_job=${qkud_job%%;*}

printf 'QRTE delta_t=0.05 job: %s\n' "${qrte_job}"
printf 'QKUD epsilon=0.05 job: %s\n' "${qkud_job}"
printf 'Excluded molecule: Ferrocene_ceo\n'
printf 'QRTE output: runs/qrte_natorb_dt0.05\n'
printf 'QKUD output: runs/qkud_natorb_eps0.05\n'
