#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_feno_dt0p01_to_0p10_100iter_array.sbatch")
job=${job%%;*}

printf 'QRTE sfX2C1eROHF feno_2plus dt=0.01..0.10, 100 iterations array job: %s\n' "${job}"
printf 'Array tasks: 0-9 -> dt=0.01,0.02,...,0.10\n'
printf 'Output root: runs/qrte_sfx2c1erohf_natorb_feno_dt0.01_to_0.10_100iter\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf\n'
printf 'Reference cache target: d05ff05948743c30710e6737104a6b7f861c578e089df5d59733ebcba4156039.npz\n'
