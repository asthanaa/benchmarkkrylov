#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /abs/path/to/case_dir" >&2
  exit 2
fi

ROOT="/home/ayush.asthana/code/CompressK"
QP_ROOT="/home/ayush.asthana/quantum_package/qp2"
CASE_DIR="$(python - <<'PY' "$1"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
INPUT_JSON="$CASE_DIR/input.json"

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "missing input.json: $INPUT_JSON" >&2
  exit 2
fi

RUN_NAME="$(python - <<'PY' "$INPUT_JSON"
import json
import sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
runs = spec.get('runs', [])
if not runs:
    raise SystemExit('input.json does not define any runs')
print(str(runs[0]['name']))
PY
)"

SBATCH_PATH="$CASE_DIR/submit_polyene_ckci_n_cachedref.sbatch"
STDOUT_PATH="$CASE_DIR/slurm-%j.out"
STDERR_PATH="$CASE_DIR/slurm-%j.err"
JOB_NAME="poly24_ckcin"
MEMORY_GB="170G"
CPUS="24"
TIME_LIMIT="10-00:00:00"
TOPN="512"
MAX_N="2"

echo "Syncing and rebuilding the installed QP CCIK plugin so the final-support snapshot writer is available..."
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
python "$ROOT/scripts/advanced/qp_ccik_bridge.py" rebuild-plugin --repo "$ROOT/ccik_qp/stable/ccik"

PREFERRED_NODE="$(
  sinfo -N -p talon -h -o '%N %t' \
  | awk '$2=="idle" && $1!="talon20" {print $1; exit}'
)"

if [[ -n "$PREFERRED_NODE" ]]; then
  NODE_DIRECTIVE="#SBATCH --nodelist=$PREFERRED_NODE"
else
  NODE_DIRECTIVE="#SBATCH --exclude=talon20"
fi

cat > "$SBATCH_PATH" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=talon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEMORY_GB
#SBATCH --time=$TIME_LIMIT
$NODE_DIRECTIVE
#SBATCH --chdir=$ROOT
#SBATCH --output=$STDOUT_PATH
#SBATCH --error=$STDERR_PATH

set -euo pipefail

ROOT="$ROOT"
QP_ROOT="$QP_ROOT"
CASE_DIR="$CASE_DIR"
INPUT_JSON="$INPUT_JSON"
RUN_NAME="$RUN_NAME"

bash "\$ROOT/scripts/advanced/run_polyene_thick_slurm.sh" "\$INPUT_JSON"

source "\$ROOT/scripts/advanced/qp_slurm_env.sh"
set +e
set +u
set +o pipefail
source "\$QP_ROOT/quantum_package.rc" >/dev/null 2>&1
rc=\$?
set -o pipefail
set -u
set -e
if [[ \$rc -ne 0 ]]; then
  echo "failed to source \$QP_ROOT/quantum_package.rc (rc=\$rc)" >&2
  exit "\$rc"
fi
source "\$ROOT/.venv/bin/activate"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cka_export_qp_maxmem_from_slurm

python "\$ROOT/scripts/advanced/polyene_ckci_n_refine.py" \\
  --case "\$CASE_DIR" \\
  --run "\$RUN_NAME" \\
  --topn "$TOPN" \\
  --max-n "$MAX_N"
EOF

chmod +x "$SBATCH_PATH"

echo "Submitting $SBATCH_PATH"
sbatch "$SBATCH_PATH"
