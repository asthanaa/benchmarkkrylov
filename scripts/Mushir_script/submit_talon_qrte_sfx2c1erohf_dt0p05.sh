#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

feno_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p05_feno_2plus.sbatch")
fenoco3_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p05_fenoco3_minus.sbatch")
feno_job=${feno_job%%;*}
fenoco3_job=${fenoco3_job%%;*}

printf 'QRTE sfX2C1eROHF feno_2plus dt=0.05 job: %s\n' "${feno_job}"
printf 'QRTE sfX2C1eROHF fenoco3_minus dt=0.05 job: %s\n' "${fenoco3_job}"
printf 'Output: runs/qrte_sfx2c1erohf_natorb_dt0.05\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf\n'
