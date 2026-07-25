#!/bin/bash
#SBATCH --job-name=mushir-n2-spin2-qrte-otol
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

mkdir -p logs

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

exec "${PYTHON}" scripts/Mushir_script/run_n2_molvqe_spin2_qrte_overlap_tol_scan.py \
  --case-id N2_equi_cc_pVTZ_2 \
  --outdir runs/qrte_natorb_dt0.05_n2_equi_spin2_overlap_tol_scan \
  --summary-csv runs/qrte_natorb_dt0.05_n2_equi_spin2_overlap_tol_scan/overlap_tol_scan_summary.csv \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --delta-t "${QCANT_DELTA_T:-0.05}" \
  --overlap-tol 1e-6 \
  --overlap-tol 1e-7 \
  --overlap-tol 1e-8 \
  --overlap-tol 1e-9 \
  --overlap-tol 1e-10 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-2}" \
  --hamiltonian-source casscf_natorb \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb
