#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs

job=$(sbatch --parsable "${SCRIPT_DIR}/run_rohf_molvqe_fe_cases.sbatch")
job=${job%%;*}

printf 'ROHF/CASSCF MolVQE FeO + feo4_2minus job: %s\n' "${job}"
printf 'Output: runs/rohf_molvqe_fe_cases\n'
