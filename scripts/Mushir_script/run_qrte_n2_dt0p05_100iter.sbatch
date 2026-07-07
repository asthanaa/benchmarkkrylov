#!/bin/bash
#SBATCH --job-name=mushir-qrte-n2-dt0.05-100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

exec "${PYTHON}" scripts/run_krylov_pool.py \
  --method qrte \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --case-id N2_equi_cc_pVTZ_2 \
  --outdir runs/qrte_natorb_dt0.05_100iter \
  --n-steps 100 \
  --delta-t 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --concurrency 1 \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb \
  --early-stop-chemical-accuracy \
  --order manifest
