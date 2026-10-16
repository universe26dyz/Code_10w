#!/usr/bin/env bash
# Shared R0 native-map cache resolution for R1 fit-first runners.
phase4r_resolve_r0_native_maps() {
  local sid=${1:?subject is required}
  local run_id=${2:?R1 run ID is required}
  if [[ -n ${PHASE4R_R0_NATIVE_MAPS_ROOT:-} ]]; then
    printf '%s\n' "$PHASE4R_R0_NATIVE_MAPS_ROOT"
  elif [[ -n ${PHASE4R_R0_CACHE_ROOT:-} ]]; then
    printf '%s/%s/native_maps\n' "$PHASE4R_R0_CACHE_ROOT" "$sid"
  else
    local r0_run_id=${PHASE4R_R0_RUN_ID:-$run_id}
    printf '%s/Code_10w_runs/paper_joint_vs_2dfit/real/R0_full/%s/fit_first/%s/native_maps\n' "$DATA" "$r0_run_id" "$sid"
  fi
}
