#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

dt0p1_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p1_feno_2plus.sbatch")
dt0p2_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p2_feno_2plus.sbatch")
dt0p5_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_dt0p5_feno_2plus.sbatch")
dt0p1_job=${dt0p1_job%%;*}
dt0p2_job=${dt0p2_job%%;*}
dt0p5_job=${dt0p5_job%%;*}

printf 'QRTE sfX2C1eROHF feno_2plus dt=0.1 job: %s\n' "${dt0p1_job}"
printf 'QRTE sfX2C1eROHF feno_2plus dt=0.2 job: %s\n' "${dt0p2_job}"
printf 'QRTE sfX2C1eROHF feno_2plus dt=0.5 job: %s\n' "${dt0p5_job}"
printf 'Outputs:\n'
printf '  runs/qrte_sfx2c1erohf_natorb_dt0.1\n'
printf '  runs/qrte_sfx2c1erohf_natorb_dt0.2\n'
printf '  runs/qrte_sfx2c1erohf_natorb_dt0.5\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf\n'
