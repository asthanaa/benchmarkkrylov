#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ayush.asthana/code/CompressK"
QP_ROOT="/home/ayush.asthana/quantum_package/qp2"
OPENBLAS_LIB="/cm/shared/apps/spack/opt/spack/linux-rhel9-icelake/gcc-11.3.1/openblas-0.3.27-tjsex4shr6vfowj3tgp23jyvftvaxl2q/lib"
SCRIPT_REL="scripts/advanced/rebase_mo2_reference.py"
LOG_DIR="$ROOT/inputs/mo2/1.94/slurm"
JOB_NAME="mo2_ref_rebase"
PROBE_INPUT="$ROOT/inputs/mo2/1.94/mo2_cas12e24_R1p94_ccpvtzdk_dmrg1000_exactized_m031_k3000_s1000000_r03_t12/input.json"

mkdir -p "$LOG_DIR"

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(--all-mo2-cases)
fi

idle_talon_node="$(sinfo -N -h -p talon -t idle -o '%N' | tr ' ' '\n' | sed '/^$/d' | head -n 1)"
PARTITION="talon"
if [[ -n "${idle_talon_node:-}" ]]; then
  NODE_SPEC=(--nodelist="$idle_talon_node")
else
  NODE_SPEC=()
fi

quoted_args=()
for arg in "${ARGS[@]}"; do
  printf -v escaped '%q' "$arg"
  quoted_args+=("$escaped")
done
joined_args="${quoted_args[*]}"

wrap_body=$(
  cat <<EOF
cd '$ROOT'
source '$ROOT/scripts/advanced/qp_slurm_env.sh'
set +e
set +u
set +o pipefail
source '$QP_ROOT/quantum_package.rc' >/dev/null 2>&1
rc=\$?
set -o pipefail
set -u
set -e
if [[ \$rc -ne 0 ]]; then
  echo "failed to source $QP_ROOT/quantum_package.rc (rc=\$rc)" >&2
  exit "\$rc"
fi
export LD_LIBRARY_PATH='$OPENBLAS_LIB':\${LD_LIBRARY_PATH:-}
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cka_export_qp_maxmem_from_slurm
echo "[stage] export-only probe: $PROBE_INPUT"
python '$SCRIPT_REL' --input '$PROBE_INPUT' --export-only --force-reference
echo "[stage] full rebase: $joined_args"
python '$SCRIPT_REL' $joined_args
EOF
)
printf -v wrap_cmd 'bash -lc %q' "$wrap_body"

job_id="$(
  sbatch --parsable \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    "${NODE_SPEC[@]}" \
    --time=7-00:00:00 \
    --cpus-per-task=12 \
    --mem=256G \
    --output="$LOG_DIR/${JOB_NAME}_%j.out" \
    --error="$LOG_DIR/${JOB_NAME}_%j.err" \
    --wrap "$wrap_cmd"
)"

echo "submitted job $job_id"
echo "partition=$PARTITION"
if [[ ${#NODE_SPEC[@]} -gt 0 ]]; then
  echo "node=${NODE_SPEC[0]#--nodelist=}"
else
  echo "node=unconstrained"
fi
