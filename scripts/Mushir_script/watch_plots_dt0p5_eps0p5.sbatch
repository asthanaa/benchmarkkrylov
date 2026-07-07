#!/usr/bin/env bash
#SBATCH --job-name=mushir-live-0p5
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15-00:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

WATCH_JOB_ARGS=()
for job_id in ${WATCH_JOB_IDS:-}; do
  WATCH_JOB_ARGS+=(--watch-job-id "${job_id}")
done

exec "${PYTHON}" scripts/Mushir_script/watch_dt0p5_eps0p5_plots.py \
  --method both \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --plot-dir plots/mushir_dt0p5_eps0p5 \
  --interval-seconds "${MUSHIR_PLOT_INTERVAL_SECONDS:-60}" \
  --status-path plots/mushir_dt0p5_eps0p5/live_plot_watch_status.json \
  --allow-empty-refresh \
  "${WATCH_JOB_ARGS[@]}"
