#!/bin/bash
#SBATCH --job-name=mushir-qrte-cas-piW-exact
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.out
#SBATCH --error=/home/mushir.kolathingal/benchmarkkrylov/logs/%x-%A_%a.err

set -euo pipefail
cd /home/mushir.kolathingal/benchmarkkrylov

# Slurm runs a spool copy, so resolve the driver from the shared repository.
# Reuse the validated CAS workflow, changing only propagation and output location.
export QCANT_EVOLUTION_MODE=sparse
export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_casscf_natorb_pi_over_w_small10_exact}"
exec bash scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10_array.sbatch
