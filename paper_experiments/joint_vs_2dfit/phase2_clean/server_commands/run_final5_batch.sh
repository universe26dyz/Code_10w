#!/usr/bin/env bash
# Thin server-only forwarder; Python final5_batch_runner is the authority.
set -euo pipefail

[[ $# -eq 6 ]] || { echo "usage: $0 plan|execute|metrics|status|validate REGISTRY FINAL_ROOT DEVICE PYTHON NESVOR" >&2; exit 2; }
COMMAND=$1
REGISTRY=$(realpath -m "$2")
FINAL_ROOT=$(realpath -m "$3")
DEVICE=$4
PYTHON=$5
NESVOR=$(realpath -m "$6")
case "$COMMAND" in plan|execute|metrics|status|validate) ;; *) echo "invalid command: $COMMAND" >&2; exit 2;; esac
test -f "$REGISTRY"; test -x "$PYTHON"; test -x "$NESVOR"
REPO=${REPO:-/data/dengyz/code/Code_10w}
cd "$REPO"
exec "$PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.final5_batch_runner "$COMMAND" --registry "$REGISTRY" --final-root "$FINAL_ROOT" --device "$DEVICE" --python "$PYTHON" --nesvor "$NESVOR"
