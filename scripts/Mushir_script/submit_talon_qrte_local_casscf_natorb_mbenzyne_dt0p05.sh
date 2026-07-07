#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

singlet_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_local_casscf_natorb_dt0p05_mbenzyne_singlet.sbatch")
spin2_job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_local_casscf_natorb_dt0p05_mbenzyne_spin2.sbatch")
singlet_job=${singlet_job%%;*}
spin2_job=${spin2_job%%;*}

printf 'QRTE m-Benzyne singlet local CASSCF/natorb dt=0.05 job: %s\n' "${singlet_job}"
printf 'QRTE m-Benzyne spin-2 local CASSCF/natorb dt=0.05 job: %s\n' "${spin2_job}"
printf 'Output: runs/qrte_mbenzyne_local_casscf_natorb_dt0.05\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_mbenzyne_localref\n'
