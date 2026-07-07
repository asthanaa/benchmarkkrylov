#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p05_fenoco3_minus_unconverged.sbatch")
job=${job%%;*}

printf 'QRTE sfX2C1eROHF fenoco3_minus unconverged CASSCF/natorb dt=0.05 job: %s\n' "${job}"
printf 'Output: runs/qrte_sfx2c1erohf_natorb_unconverged_dt0.05\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf_unconverged\n'
