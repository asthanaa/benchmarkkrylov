#!/bin/bash
#SBATCH --job-name=mushir-qkud-fe-eps0.05
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

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
CASE_IDS=(FeO fenoco3_minus feno_2plus)
CASE_ID=${CASE_IDS[${SLURM_ARRAY_TASK_ID}]}

exec "${PYTHON}" scripts/Mushir_script/run_qkud_fe_eps0p05_cases.py \
  --case-id "${CASE_ID}" \
  --outdir runs/qkud_fe_eps0.05 \
  --n-steps "${QCANT_N_STEPS:-100}" \
  --epsilon 0.05 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --rohf-hamiltonian-cache-dir hamiltonian_cache/casscf_rohf_molvqe_fe_cases \
  --sfx-hamiltonian-cache-dir hamiltonian_cache/casscf_natorb_sfx2c1erohf \
  --sfx-unconverged-hamiltonian-cache-dir hamiltonian_cache/casscf_natorb_sfx2c1erohf_unconverged \
  --rohf-scf-conv-tol 1e-10 \
  --rohf-scf-diis-space 12 \
  --rohf-scf-level-shift 0.0 \
  --rohf-scf-max-cycle 200 \
  --rohf-casscf-conv-tol 1e-10 \
  --rohf-casscf-max-cycle "${QCANT_CASSCF_MAX_CYCLE:-300}" \
  --early-stop-chemical-accuracy
