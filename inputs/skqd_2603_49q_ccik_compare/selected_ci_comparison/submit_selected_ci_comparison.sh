#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ayush.asthana/code/CompressK/inputs/skqd_2603_49q_ccik_compare/selected_ci_comparison"
REPO="/home/ayush.asthana/code/CompressK"
MODE="${1:-all}"
CPUS=24

mkdir -p "${ROOT}/logs"

FREE_ON_TALON04="$(python3 - <<'PY'
import re
import subprocess

try:
    text = subprocess.check_output(["scontrol", "show", "node", "talon04"], text=True)
    alloc = int(re.search(r"CPUAlloc=(\d+)", text).group(1))
    total = int(re.search(r"CPUTot=(\d+)", text).group(1))
    print(total - alloc)
except Exception:
    print(0)
PY
)"

PARTITION="talon"
NODELIST=""

if [[ "${FREE_ON_TALON04}" -ge "${CPUS}" ]]; then
  PARTITION="talon-ayush"
  NODELIST="talon04"
else
  IDLE_NODE="$(sinfo -N -p talon -h -o '%N %t' | awk '$2 == "idle" {print $1; exit}')"
  if [[ -n "${IDLE_NODE}" ]]; then
    NODELIST="${IDLE_NODE}"
  fi
fi

JOB_NAME="skqd_hci_cipsi_${MODE}"
SBATCH_ARGS=(
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --nodes=1
  --ntasks=1
  --cpus-per-task="${CPUS}"
  --mem=256G
  --time=7-00:00:00
  --chdir="${REPO}"
  --output="${ROOT}/logs/%x-%j.out"
  --error="${ROOT}/logs/%x-%j.err"
)

if [[ -n "${NODELIST}" ]]; then
  SBATCH_ARGS+=(--nodelist="${NODELIST}")
fi

WRAP_CMD="export OMP_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} NUMEXPR_NUM_THREADS=${CPUS}; python3 ${ROOT}/selected_ci_comparison.py --mode ${MODE}"
sbatch "${SBATCH_ARGS[@]}" --wrap "${WRAP_CMD}"
