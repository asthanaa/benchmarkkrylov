#!/usr/bin/env bash
#SBATCH --job-name=qcant-param-probe
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --exclusive
#SBATCH --time=28-00:00:00
#SBATCH --array=0-7%8
#SBATCH --exclude=talon06,talon07,talon12,talon13,talon14
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
  --values 0.5,1.0
  --n-steps 50
  --delta-t 0.1
  --epsilon 0.1
  --evolution-mode trotter
  --trotter-steps 1
  --concurrency "${QCANT_PROBE_CONCURRENCY:-6}"
  --max-workers "${QCANT_PROBE_MAX_WORKERS:-6}"
  --hamiltonian-source casscf_natorb
  --hamiltonian-cache-dir hamiltonian_cache/casscf_natorb
)

if (( SLURM_ARRAY_TASK_ID < 4 )); then
  METHOD=qrte
  BASELINE_DIR=runs/qrte_natorb_dt0.1
  OUTDIR=runs/qrte_natorb_delta_sweep
  SHARD_INDEX="${SLURM_ARRAY_TASK_ID}"
  CASE_ARGS=(
    --case-id cyclobutadiene_ceo
    --case-id CO_equi_cc_pVTZ_0
    --case-id CH2O_equi_cc_pVTZ_0
    --case-id m_Benzyne_equi_cc_pVTZ_2
    --case-id m_Benzyne_equi_cc_pVTZ_0
    --case-id Ferrocene_ceo
    --case-id feo4_2minus_ceo
    --case-id fenoco3_minus
    --case-id FeO
    --case-id feno_2plus
  )
else
  METHOD=qkud
  BASELINE_DIR=runs/qkud_natorb_eps0.1
  OUTDIR=runs/qkud_natorb_epsilon_sweep
  SHARD_INDEX="$((SLURM_ARRAY_TASK_ID - 4))"
  CASE_ARGS=(
    --case-id cyclobutadiene_ceo
    --case-id m_Benzyne_equi_cc_pVTZ_2
    --case-id m_Benzyne_equi_cc_pVTZ_0
    --case-id Ferrocene_ceo
    --case-id feo4_2minus_ceo
    --case-id fenoco3_minus
    --case-id FeO
    --case-id feno_2plus
  )
fi

exec "${PYTHON}" scripts/run_krylov_parameter_sweep.py \
  --method "${METHOD}" \
  --baseline-dir "${BASELINE_DIR}" \
  --outdir "${OUTDIR}" \
  --shard-count 4 \
  --shard-index "${SHARD_INDEX}" \
  "${COMMON_ARGS[@]}" \
  "${CASE_ARGS[@]}"
