#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs plots/mushir_dt0p5_eps0p5

qrte_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_dt0p5_pool.sbatch")
qkud_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qkud_eps0p5_pool.sbatch")
qrte_job=${qrte_job%%;*}
qkud_job=${qkud_job%%;*}

export WATCH_JOB_IDS="${qrte_job} ${qkud_job}"
watch_job=$(sbatch --parsable \
  --export=ALL,WATCH_JOB_IDS \
  "${SCRIPT_DIR}/watch_plots_dt0p5_eps0p5.sbatch")
watch_job=${watch_job%%;*}

printf 'QRTE delta_t=0.5 job: %s\n' "${qrte_job}"
printf 'QKUD epsilon=0.5 job: %s\n' "${qkud_job}"
printf 'Live plot watcher job: %s\n' "${watch_job}"
printf 'QRTE output: runs/qrte_natorb_dt0.5\n'
printf 'QKUD output: runs/qkud_natorb_eps0.5\n'
printf 'Plot output: plots/mushir_dt0p5_eps0p5\n'
