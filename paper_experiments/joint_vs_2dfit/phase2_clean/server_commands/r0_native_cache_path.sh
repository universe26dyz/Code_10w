#!/usr/bin/env bash
# Shared immutable R0 native-map cache resolution for all R1 consumers.
phase4r_resolve_r0_native_maps() {
  local sid=${1:?subject is required}
  local run_id=${2:?R1 run ID is required}
  if [[ -n ${PHASE4R_R0_NATIVE_MAPS_ROOT:-} ]]; then
    printf '%s\n' "$PHASE4R_R0_NATIVE_MAPS_ROOT"
  elif [[ -n ${PHASE4R_R0_CACHE_ROOT:-} ]]; then
    printf '%s/%s/native_maps\n' "$PHASE4R_R0_CACHE_ROOT" "$sid"
  else
    local cache_id=${PHASE4R_R0_CACHE_ID:-R0_PHASE4R}
    printf '%s/Code_10w_runs/paper_joint_vs_2dfit/real/R0_native_cache/%s/%s/native_maps\n' "$DATA" "$cache_id" "$sid"
  fi
}
