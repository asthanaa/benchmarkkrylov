#!/usr/bin/env bash
#SBATCH --job-name=qcant-missing-plot36
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=28-00:00:00
#SBATCH --array=0-5%6
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

case "${SLURM_ARRAY_TASK_ID}" in
  0) METHOD=qrte; CASE_ID=Ferrocene_ceo; OUTDIR=runs/qrte_natorb_dt0.1; METHOD_ARG=(--delta-t 0.1) ;;
  1) METHOD=qrte; CASE_ID=fenoco3_minus; OUTDIR=runs/qrte_natorb_dt0.1; METHOD_ARG=(--delta-t 0.1) ;;
  2) METHOD=qrte; CASE_ID=magnesium_porphyrin_ceo; OUTDIR=runs/qrte_natorb_dt0.1; METHOD_ARG=(--delta-t 0.1) ;;
  3) METHOD=qkud; CASE_ID=Ferrocene_ceo; OUTDIR=runs/qkud_natorb_eps0.1; METHOD_ARG=(--epsilon 0.1) ;;
  4) METHOD=qkud; CASE_ID=fenoco3_minus; OUTDIR=runs/qkud_natorb_eps0.1; METHOD_ARG=(--epsilon 0.1) ;;
  5) METHOD=qkud; CASE_ID=magnesium_porphyrin_ceo; OUTDIR=runs/qkud_natorb_eps0.1; METHOD_ARG=(--epsilon 0.1) ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

exec "${PYTHON}" scripts/run_krylov_benchmark.py \
  --method "${METHOD}" \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --outdir "${OUTDIR}" \
  --case-id "${CASE_ID}" \
  --n-steps "${QCANT_N_STEPS:-50}" \
  "${METHOD_ARG[@]}" \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-36}" \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --early-stop-chemical-accuracy
