#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/environment.dmrg.yml"
ENV_PREFIX="${1:-${ROOT_DIR}/.conda/cka-dmrg-py312}"

if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="mamba"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="conda"
else
  echo "Missing conda-compatible package manager. Install conda or mamba first." >&2
  exit 2
fi

mkdir -p "$(dirname "${ENV_PREFIX}")"

if [ -d "${ENV_PREFIX}" ]; then
  "${CONDA_BIN}" env update --prefix "${ENV_PREFIX}" --file "${ENV_FILE}" --prune
else
  "${CONDA_BIN}" env create --prefix "${ENV_PREFIX}" --file "${ENV_FILE}"
fi

"${ENV_PREFIX}/bin/python" - <<'PY'
import block2
import pyscf
print("DMRG bootstrap OK")
print("block2", getattr(block2, "__version__", "<unknown>"))
print("pyscf", pyscf.__version__)
PY

cat <<EOF

Environment ready at:
  ${ENV_PREFIX}

To use it in study inputs, either:
  1. omit study.dmrg_python and let the runner use ${ROOT_DIR}/.conda/cka-dmrg-py312/bin/python
  2. or set study.dmrg_python to:
     ${ENV_PREFIX}/bin/python
EOF
