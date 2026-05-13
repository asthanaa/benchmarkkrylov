#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

qrte_job=$(sbatch --parsable run_qrte_dt0.1_pool.sbatch)
qkud_job=$(sbatch --parsable run_qkud_eps0.1_pool.sbatch)
qrte_job=${qrte_job%%;*}
qkud_job=${qkud_job%%;*}
plot_job=$(sbatch --parsable --dependency=afterany:${qrte_job}:${qkud_job} make_plots.sbatch)
plot_job=${plot_job%%;*}

printf 'QRTE job: %s\n' "$qrte_job"
printf 'QKUD job: %s\n' "$qkud_job"
printf 'Plot job: %s\n' "$plot_job"
