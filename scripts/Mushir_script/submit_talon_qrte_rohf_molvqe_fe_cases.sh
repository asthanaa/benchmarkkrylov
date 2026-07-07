#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_rohf_molvqe_fe_cases.sbatch")
job=${job%%;*}

printf 'QRTE ROHF MolVQE FeO + feo4_2minus dt=0.05,0.1 job: %s\n' "${job}"
printf 'Output: runs/qrte_rohf_molvqe_fe_cases\n'
printf 'Plots: plots/qrte_rohf_molvqe_fe_cases\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_rohf_molvqe_fe_cases\n'
