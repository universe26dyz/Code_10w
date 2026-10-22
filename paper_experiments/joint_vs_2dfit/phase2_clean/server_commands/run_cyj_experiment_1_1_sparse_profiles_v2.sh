#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" ]] || { echo "only --dry-run is authorized; no profile or reconstruction output may be written" >&2; exit 2; }
cat <<'PLAN'
dry-run: inspect each profile JSON and print only future commands.
Future profile command: python -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse --profile <profile.json> --fold <n> --prepared-root <root> --output-root <new-root> --dry-run
Future training inputs: only stacks declared training_stack_present; `none` has no observations.npz.
Future outputs (fresh only): Proposed, Fit-first T1/T2, poses, central/psf128 endpoints, native bundles, masks, metrics, figures, summary.
PLAN
