#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_qrte_rohf_feo_dt0p2_0p5_2p0_100iter_array.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_qrte_rohf_feo_dt0p2_0p5_2p0_100iter_jobid.txt

printf 'QRTE ROHF MolVQE FeO dt=0.2,0.5,2.0 100-iteration array job: %s\n' "${job}"
printf 'Array tasks: 0 -> dt=0.2, 1 -> dt=0.5, 2 -> dt=2.0\n'
printf 'Output root: runs/qrte_rohf_molvqe_fe_cases_100iter/FeO\n'
printf 'Plots: plots/qrte_rohf_molvqe_fe_cases_100iter\n'
printf 'Hamiltonian cache: hamiltonian_cache/casscf_rohf_molvqe_fe_cases\n'
