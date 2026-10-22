#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || { echo "usage: $0 [--dry-run|--execute] ..." >&2; exit 2; }
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PHASE4R_PYTHON:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
if [[ $# -eq 7 ]]; then
  MANIFEST=$2
  B6_MODEL=$3
  DECODER=$4
  OUTPUT=$5
  GENERATOR_SHA=$6
  MANIFEST_SHA=$7
  # The Python entry validates the immutable manifest and prints the exact
  # future Q002S640 command.  Dry-run never starts training.
  test -d "$REPO" && test -x "$PY" && test -f "$MANIFEST" && test -f "$B6_MODEL" && test -f "$DECODER"
  test ! -e "$OUTPUT" || { echo "refusing existing output: $OUTPUT" >&2; exit 1; }
  cd "$REPO"
  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.experiment_1_1 \
    --sparse-manifest "$MANIFEST" --b6-model "$B6_MODEL" --decoder "$DECODER" --output "$OUTPUT" --expected-generator-git-sha "$GENERATOR_SHA" --expected-manifest-sha256 "$MANIFEST_SHA" ${MODE/--dry-run/}
  exit 0
fi
[[ $# -eq 1 ]] || { echo "usage: $0 [--dry-run|--execute] [sparse_manifest b6_model decoder fresh_output generator_git_sha manifest_sha256]" >&2; exit 2; }
cat <<'PLAN'
dry-run: inspect each profile JSON and print only future commands.
Concrete dry-run: run_cyj_experiment_1_1_sparse_profiles_v2.sh --dry-run <sparse_manifest> <b6_model> <decoder> <fresh_output> <generator_git_sha> <manifest_sha256>
Future profile command: python -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse --profile <profile.json> --fold <n> --prepared-root <root> --output-root <new-root> --dry-run
Future training inputs: only stacks declared training_stack_present; `none` has no observations.npz.
Future outputs (fresh only): Proposed, Fit-first T1/T2, poses, central/psf128 endpoints, native bundles, masks, metrics, figures, summary.
PLAN
