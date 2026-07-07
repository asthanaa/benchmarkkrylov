#!/bin/bash
#SBATCH --job-name=mushir-qrte-x2c-ferrocene-dt0.05
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%j.err

set -euo pipefail

cd /home/mushir.kolathingal/benchmarkkrylov

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-mushir.kolathingal}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"

mkdir -p logs "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

exec "${PYTHON}" scripts/Mushir_script/run_qrte_sfx2c1erohf_casscf_ferrocene_dt0p05.py \
  --outdir runs/qrte_sfx2c1erohf_casscf_ferrocene_dt0.05 \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --delta-t 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_sfx2c1erohf_ferrocene \
  --casscf-conv-tol 1e-10 \
  --casscf-max-cycle "${QCANT_CASSCF_MAX_CYCLE:-300}" \
  --casscf-match-tolerance "${QCANT_CASSCF_MATCH_TOL:-1e-5}" \
  --early-stop-chemical-accuracy
