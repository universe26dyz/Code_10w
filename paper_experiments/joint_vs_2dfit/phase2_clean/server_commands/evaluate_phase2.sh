#!/usr/bin/env bash
set -euo pipefail
# not auto-launched: evaluation starts only when an operator runs this script.
"$(dirname "$0")/server_preflight_phase2.sh" >/tmp/phase2_preflight.json
REPO=${REPO:-/data/dengyz/code/Code_10w}; cd "$REPO"; python -m paper_experiments.joint_vs_2dfit.phase2_clean.evaluation --contract paper_experiments/joint_vs_2dfit/phase2_clean/phase2_input_manifest.json
