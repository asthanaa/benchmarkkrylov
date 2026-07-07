#!/bin/bash
#SBATCH --job-name=mushir-qkud-casfix-eps0.05
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=120G
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-mushir.kolathingal}}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"

mkdir -p "${MPLCONFIGDIR}" "${PYTHONPYCACHEPREFIX}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

exec "${PYTHON}" scripts/Mushir_script/run_qkud_corrected_active_space_eps0p05_cases.py \
  --manifest manifests/molvqe21_fig1b_manifest.csv \
  --active-orbital-overrides manifests/molvqe21_active_orbital_overrides.json \
  --case-id cyclobutadiene_ceo \
  --case-id m_Benzyne_equi_cc_pVTZ_0 \
  --case-id m_Benzyne_equi_cc_pVTZ_2 \
  --outdir runs/qkud_corrected_active_space_eps0.05 \
  --n-steps "${QCANT_N_STEPS:-80}" \
  --epsilon 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-36}" \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb_corrected_active_space_dt0p05 \
  --casscf-conv-tol 1e-10 \
  --casscf-max-cycle 200 \
  --casscf-match-tolerance "${QCANT_CASSCF_MATCH_TOL:-1e-5}"
