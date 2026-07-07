#!/bin/bash
#SBATCH --job-name=mushir-qrte-rohf-feo-dtgrid-100
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=92G
#SBATCH --time=12:00:00
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
export MPLCONFIGDIR=/tmp/matplotlib-${USER}
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/python-pycache-${USER:-mushir.kolathingal}}"

PYTHON=${PYTHON:-/home/mushir.kolathingal/miniforge3/envs/expk/bin/python}

DT_VALUES=(0.2 0.5 2.0)
DT_LABELS=(dt0p2 dt0p5 dt2p0)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DELTA_T=${DT_VALUES[${TASK_ID}]}
DT_LABEL=${DT_LABELS[${TASK_ID}]}

printf 'Running FeO QRTE ROHF/MolVQE 100-iteration task %s: %s\n' "${TASK_ID}" "${DT_LABEL}"

exec "${PYTHON}" scripts/Mushir_script/run_qrte_rohf_molvqe_fe_cases.py \
  --case-id FeO \
  --delta-t "${DELTA_T}" \
  --outdir runs/qrte_rohf_molvqe_fe_cases_100iter \
  --plot-dir plots/qrte_rohf_molvqe_fe_cases_100iter \
  --n-steps 100 \
  --evolution-mode trotter \
  --trotter-steps 1 \
  --max-workers "${QCANT_MAX_WORKERS:-18}" \
  --hamiltonian-cache-dir hamiltonian_cache/casscf_rohf_molvqe_fe_cases \
  --scf-conv-tol 1e-10 \
  --scf-diis-space 12 \
  --scf-level-shift 0.0 \
  --scf-max-cycle 200 \
  --casscf-conv-tol 1e-10 \
  --casscf-max-cycle "${QCANT_CASSCF_MAX_CYCLE:-300}" \
  --newton-if-not-converged
