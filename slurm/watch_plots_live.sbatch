#!/usr/bin/env bash
#SBATCH --job-name=qcant-live-plots
#SBATCH --partition=talon
#SBATCH --exclude=talon04
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

"${PYTHON}" scripts/watch_and_update_plots.py \
  --method both \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --plot-dir plots \
  --interval-seconds 60 \
  "${WATCH_JOB_ARGS[@]}"
