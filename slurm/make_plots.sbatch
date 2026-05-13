#!/usr/bin/env bash
#SBATCH --job-name=qcant-krylov-plots
#SBATCH --partition=talon-ayush
#SBATCH --nodelist=talon04
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=/home/ayush.asthana/code/benchmark/logs/%x-%j.out
#SBATCH --error=/home/ayush.asthana/code/benchmark/logs/%x-%j.err

set -euo pipefail

cd /home/ayush.asthana/code/benchmark

PYTHON=${PYTHON:-/home/ayush.asthana/miniconda3/bin/python}

"${PYTHON}" scripts/collect_and_plot.py \
  --method both \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --plot-dir plots
