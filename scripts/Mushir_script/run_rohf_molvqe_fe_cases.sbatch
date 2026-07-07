#!/bin/bash
#SBATCH --job-name=mushir-rohf-molvqe-fe
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=/home/mushir.kolathingal/miniforge3/envs/expk/bin/python

exec "${PYTHON}" scripts/Mushir_script/run_rohf_molvqe_fe_cases.py \
  --case-id FeO \
  --case-id feo4_2minus_ceo \
  --outdir runs/rohf_molvqe_fe_cases \
  --scf-conv-tol 1e-10 \
  --scf-diis-space 12 \
  --scf-level-shift 0.0 \
  --scf-max-cycle 200 \
  --casscf-conv-tol 1e-10 \
  --casscf-max-cycle 100 \
  --newton-if-not-converged
