#!/usr/bin/env bash

_cka_slurm_mem_to_mib() {
  local raw="${1:-}"
  local value unit
  if [[ -z "$raw" || "$raw" == "0" || "$raw" == "UNLIMITED" ]]; then
    echo 0
    return 0
  fi
  if [[ "$raw" =~ ^([0-9]+)([KMGTPkmgpt]?)$ ]]; then
    value="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    case "$unit" in
      ""|M|m) echo "$value" ;;
      K|k) echo $(( value / 1024 )) ;;
      G|g) echo $(( value * 1024 )) ;;
      T|t) echo $(( value * 1024 * 1024 )) ;;
      P|p) echo $(( value * 1024 * 1024 * 1024 )) ;;
      *) echo 0 ;;
    esac
    return 0
  fi
  echo 0
}

cka_export_qp_maxmem_from_slurm() {
  local alloc_mib=0
  local alloc_gib=0
  local raw="${SLURM_MEM_PER_NODE:-}"
  local cpus="${SLURM_CPUS_PER_TASK:-1}"
  local qp_maxmem=0

  alloc_mib="$(_cka_slurm_mem_to_mib "$raw")"
  if (( alloc_mib <= 0 )); then
    raw="${SLURM_MEM_PER_CPU:-}"
    alloc_mib="$(_cka_slurm_mem_to_mib "$raw")"
    if (( alloc_mib > 0 )); then
      alloc_mib=$(( alloc_mib * cpus ))
    fi
  fi
  if (( alloc_mib <= 0 )); then
    return 0
  fi

  alloc_gib=$(( alloc_mib / 1024 ))
  if (( alloc_gib < 4 )); then
    alloc_gib=4
  fi
  qp_maxmem=$(( alloc_gib * 9 / 10 ))
  if (( qp_maxmem < 4 )); then
    qp_maxmem=4
  fi

  export QP_MAXMEM="$qp_maxmem"
  echo "QP_MAXMEM=$QP_MAXMEM (derived from Slurm allocation ~${alloc_gib} GiB)"
}
