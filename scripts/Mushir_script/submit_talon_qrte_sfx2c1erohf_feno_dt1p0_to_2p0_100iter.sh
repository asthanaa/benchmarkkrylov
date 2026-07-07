#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_sfx2c1erohf_feno_dt1p0_to_2p0_100iter_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_sfx2c1erohf_feno_dt1p0_to_2p0_100iter_jobid.txt

printf 'QRTE sfX2C1eROHF feno_2plus dt=1.00..2.00, 100 iterations array job: %s\n' "${job}"
printf 'Array tasks: 0 -> dt=1.00, 1 -> dt=1.25, 2 -> dt=1.50, 3 -> dt=1.75, 4 -> dt=2.00\n'
printf 'Output root: runs/qrte_sfx2c1erohf_natorb_feno_dt1.0_to_2.0_100iter\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_natorb_sfx2c1erohf\n'
printf 'Reference cache target: d05ff05948743c30710e6737104a6b7f861c578e089df5d59733ebcba4156039.npz\n'
