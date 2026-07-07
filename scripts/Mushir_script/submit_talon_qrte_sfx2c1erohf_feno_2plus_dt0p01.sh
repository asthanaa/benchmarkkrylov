#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p01_feno_2plus.sbatch")
job=${job%%;*}

printf 'QRTE sfX2C1eROHF feno_2plus dt=0.01 job: %s\n' "${job}"
printf 'Output: runs/qrte_sfx2c1erohf_natorb_dt0.01\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf\n'
