#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" ]] || { echo "only --dry-run is authorized; no profile or reconstruction output may be written" >&2; exit 2; }
if [[ $# -eq 5 ]]; then
  MANIFEST=$2
  B6_MODEL=$3
  DECODER=$4
  OUTPUT=$5
  # The Python entry validates the immutable manifest and prints the exact
  # future Q002S640 command.  It never starts training in this wrapper.
  "${PYTHON:-python}" -m paper_experiments.joint_vs_2dfit.phase2_clean.experiment_1_1 \
    --sparse-manifest "$MANIFEST" --b6-model "$B6_MODEL" --decoder "$DECODER" --output "$OUTPUT"
  exit 0
fi
[[ $# -eq 1 ]] || { echo "usage: $0 --dry-run [sparse_manifest b6_model decoder fresh_output]" >&2; exit 2; }
cat <<'PLAN'
dry-run: inspect each profile JSON and print only future commands.
Concrete dry-run: run_cyj_experiment_1_1_sparse_profiles_v2.sh --dry-run <sparse_manifest> <b6_model> <decoder> <fresh_output>
Future profile command: python -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse --profile <profile.json> --fold <n> --prepared-root <root> --output-root <new-root> --dry-run
Future training inputs: only stacks declared training_stack_present; `none` has no observations.npz.
Future outputs (fresh only): Proposed, Fit-first T1/T2, poses, central/psf128 endpoints, native bundles, masks, metrics, figures, summary.
PLAN
