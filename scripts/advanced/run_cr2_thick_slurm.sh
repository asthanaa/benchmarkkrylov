#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 /abs/path/to/input.json [extra study args...]" >&2
  exit 2
fi

INPUT_JSON="$1"
shift
ROOT="/home/ayush.asthana/code/CompressK"
QP_ROOT="/home/ayush.asthana/quantum_package/qp2"
OPENBLAS_LIB="/cm/shared/apps/spack/opt/spack/linux-rhel9-icelake/gcc-11.3.1/openblas-0.3.27-tjsex4shr6vfowj3tgp23jyvftvaxl2q/lib"

cd "$ROOT"
source "$ROOT/scripts/advanced/qp_slurm_env.sh"
set +e
set +u
set +o pipefail
source "$QP_ROOT/quantum_package.rc" >/dev/null 2>&1
rc=$?
set -o pipefail
set -u
set -e
if [[ $rc -ne 0 ]]; then
  echo "failed to source $QP_ROOT/quantum_package.rc (rc=$rc)" >&2
  exit "$rc"
fi
export LD_LIBRARY_PATH="$OPENBLAS_LIB:${LD_LIBRARY_PATH:-}"
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cka_export_qp_maxmem_from_slurm

python scripts/advanced/cr2_thick_iteration_study.py --input "$INPUT_JSON" "$@"
