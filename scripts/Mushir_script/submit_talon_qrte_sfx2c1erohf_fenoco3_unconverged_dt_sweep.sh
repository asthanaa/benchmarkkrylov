#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

dt0p01_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p01_fenoco3_minus_unconverged.sbatch")
dt0p01_job=${dt0p01_job%%;*}

dt0p1_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p1_fenoco3_minus_unconverged.sbatch")
dt0p1_job=${dt0p1_job%%;*}

printf 'QRTE sfX2C1eROHF fenoco3_minus unconverged CASSCF/natorb dt=0.01 job: %s\n' "${dt0p01_job}"
printf 'QRTE sfX2C1eROHF fenoco3_minus unconverged CASSCF/natorb dt=0.1 job: %s\n' "${dt0p1_job}"
printf 'Output dt=0.01: runs/qrte_sfx2c1erohf_natorb_unconverged_dt0.01\n'
printf 'Output dt=0.1:  runs/qrte_sfx2c1erohf_natorb_unconverged_dt0.1\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf_unconverged\n'
