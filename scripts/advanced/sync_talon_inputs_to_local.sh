#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote="${1:-ayush.asthana@talon.und.edu:/home/ayush.asthana/code/CompressK/inputs/}"
dest="${2:-${repo_root}/inputs/}"
max_size="${MAX_SIZE:-500m}"

common_args=(
  -av
  --max-size="${max_size}"
  --prune-empty-dirs
  --exclude='.git/'
  --exclude='*.tar.gz'
  --exclude='*.ezfio/work/'
)

if [[ "${SYNC_ALL_UNDER_LIMIT:-0}" == "1" ]]; then
  rsync "${common_args[@]}" "${remote}" "${dest}"
else
  rsync "${common_args[@]}" \
    --include='*/' \
    --include='*.json' \
    --include='*.csv' \
    --include='*.png' \
    --include='*.pdf' \
    --include='*.log' \
    --include='*.txt' \
    --include='*.md' \
    --include='*.toml' \
    --include='*.sh' \
    --include='*.tex' \
    --exclude='*' \
    "${remote}" "${dest}"
fi
