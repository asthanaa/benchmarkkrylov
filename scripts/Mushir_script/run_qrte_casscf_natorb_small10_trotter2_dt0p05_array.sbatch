#!/bin/bash
#SBATCH --job-name=mushir-qrte-cas-trotter2-dt005
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

# Resolve the driver from the shared repository, not Slurm's spool directory.
export QCANT_EVOLUTION_MODE=trotter
export QCANT_TROTTER_ORDER=2
export QCANT_DELTA_T=0.05
export QCANT_OUTDIR="${QCANT_OUTDIR:-runs/qrte_casscf_natorb_small10_trotter2_dt0p05}"
exec bash scripts/Mushir_script/run_qrte_casscf_natorb_pi_over_w_small10_array.sbatch
