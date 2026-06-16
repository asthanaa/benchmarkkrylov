#!/usr/bin/env bash
#SBATCH --job-name=qcant-missing-plot72
#SBATCH --partition=talon-large
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=28-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/home/ayush.asthana/code/benchmark/logs/%x-%A_%a.out
#SBATCH --error=/home/ayush.asthana/code/benchmark/logs/%x-%A_%a.err

set -euo pipefail

cd /home/ayush.asthana/code/benchmark

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/ayush.asthana/miniconda3/bin/python}

COMMON_ARGS=(
  --manifest manifests/molvqe21_fig1b_manifest.csv
  --n-steps "${QCANT_N_STEPS:-50}"
  --evolution-mode trotter
  --trotter-steps 1
  --concurrency "${QCANT_CONCURRENCY:-3}"
  --max-workers "${QCANT_MAX_WORKERS:-24}"
  --hamiltonian-source casscf_natorb
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb
  --early-stop-chemical-accuracy
  --order small-first
  --case-id Ferrocene_ceo
  --case-id fenoco3_minus
  --case-id magnesium_porphyrin_ceo
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    exec "${PYTHON}" scripts/run_krylov_pool.py \
      --method qrte \
      --outdir runs/qrte_natorb_dt0.1 \
      --delta-t 0.1 \
      "${COMMON_ARGS[@]}"
    ;;
  1)
    exec "${PYTHON}" scripts/run_krylov_pool.py \
      --method qkud \
      --outdir runs/qkud_natorb_eps0.1 \
      --epsilon 0.1 \
      "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac
