#!/usr/bin/env bash
#SBATCH --job-name=mushir-live-0p2
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=28-00:00:00
#SBATCH --output=/home/ayush.asthana/code/benchmark/logs/%x-%j.out
#SBATCH --error=/home/ayush.asthana/code/benchmark/logs/%x-%j.err

set -euo pipefail

cd /home/ayush.asthana/code/benchmark

PYTHON=${PYTHON:-/home/ayush.asthana/miniconda3/bin/python}

WATCH_JOB_ARGS=()
for job_id in ${WATCH_JOB_IDS:-}; do
  WATCH_JOB_ARGS+=(--watch-job-id "${job_id}")
done

exec "${PYTHON}" scripts/Mushir_script/watch_dt0p2_eps0p2_plots.py \
  --method both \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --plot-dir plots/mushir_dt0p2_eps0p2 \
  --interval-seconds "${MUSHIR_PLOT_INTERVAL_SECONDS:-60}" \
  --status-path plots/mushir_dt0p2_eps0p2/live_plot_watch_status.json \
  --allow-empty-refresh \
  "${WATCH_JOB_ARGS[@]}"
