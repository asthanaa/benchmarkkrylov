#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

cd "${BENCHMARK_ROOT}"

mkdir -p logs provenance

job=$(sbatch --parsable "${SCRIPT_DIR}/run_benchmark_qc_qrte_u2_dt0p05_overlap1e-9.sbatch")
job=${job%%;*}
printf '%s\n' "${job}" > provenance/latest_benchmark_qc_qrte_u2_dt0p05_overlap1e-9_jobid.txt

printf 'Benchmark-QC U2 QRTE dt=0.05 overlap_tol=1e-9 job: %s\n' "${job}"
printf 'Output: runs/benchmark_qc_qrte_dt0.05_overlap1e-9/U2_eq\n'
printf 'Plot after completion with:\n'
printf '  scripts/Mushir_script/plot_benchmark_qc_u2_qrte_dt0p05_convergence.py --run-dir runs/benchmark_qc_qrte_dt0.05_overlap1e-9/U2_eq --output-prefix plots/u2_benchmark_qc_qrte_dt0p05_overlap1e-9_convergence\n'
