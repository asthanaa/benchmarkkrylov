#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs plots/mushir_dt0p2_eps0p2

qrte_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_dt0p2_pool.sbatch")
qkud_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qkud_eps0p2_pool.sbatch")
qrte_job=${qrte_job%%;*}
qkud_job=${qkud_job%%;*}

watch_job=$(sbatch --parsable \
  --export=NIL \
  "${SCRIPT_DIR}/watch_plots_dt0p2_eps0p2.sbatch" \
  "${qrte_job}" "${qkud_job}")
watch_job=${watch_job%%;*}

printf 'QRTE delta_t=0.2 job: %s\n' "${qrte_job}"
printf 'QKUD epsilon=0.2 job: %s\n' "${qkud_job}"
printf 'Live plot watcher job: %s\n' "${watch_job}"
printf 'QRTE output: runs/qrte_natorb_dt0.2\n'
printf 'QKUD output: runs/qkud_natorb_eps0.2\n'
printf 'Plot output: plots/mushir_dt0p2_eps0p2\n'
