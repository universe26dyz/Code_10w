#!/usr/bin/env bash
# Immutable r5 legacy-density figure runner. Dry-run is the default.
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || {
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
}

: "${REPO:?set REPO to the checkout containing the r5 planner}"
: "${CYJ11_R5_PYTHON:?set CYJ11_R5_PYTHON}"
: "${CYJ11_R5_SUBJECT:?set CYJ11_R5_SUBJECT}"
: "${CYJ11_R5_ANALYSIS_ROOT:?set CYJ11_R5_ANALYSIS_ROOT}"
: "${CYJ11_R5_HISTORICAL_WORKTREE:?set CYJ11_R5_HISTORICAL_WORKTREE at 00d88b45}"
: "${CYJ11_R5_DIRECT_NATIVE_WORKTREE:?set CYJ11_R5_DIRECT_NATIVE_WORKTREE at d596308}"

case "$CYJ11_R5_SUBJECT" in CYJ|DYZ|HJL|HHZ|DYL_2) ;; *) echo "invalid CYJ11_R5_SUBJECT: $CYJ11_R5_SUBJECT" >&2; exit 2;; esac
RUN_LABEL=${CYJ11_R5_RUN_LABEL:-}
[[ -z "$RUN_LABEL" || "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "CYJ11_R5_RUN_LABEL must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
}

PLAN_COMMAND=(
    "$CYJ11_R5_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r5_legacy_figures
    --subject "$CYJ11_R5_SUBJECT" --analysis-root "$CYJ11_R5_ANALYSIS_ROOT"
    --historical-worktree "$CYJ11_R5_HISTORICAL_WORKTREE"
    --direct-native-worktree "$CYJ11_R5_DIRECT_NATIVE_WORKTREE"
    --python-executable "$CYJ11_R5_PYTHON" --device "${CYJ11_R5_DEVICE:-cuda:0}"
)
[[ -z ${CYJ11_R5_PREPARED_ROOT:-} ]] || PLAN_COMMAND+=(--prepared-root "$CYJ11_R5_PREPARED_ROOT")
[[ -z ${CYJ11_R5_R0_NATIVE_MAPS_ROOT:-} ]] || PLAN_COMMAND+=(--r0-native-maps-root "$CYJ11_R5_R0_NATIVE_MAPS_ROOT")
[[ -z ${CYJ11_R5_R4_NATIVE_MASK_ROOT:-} ]] || PLAN_COMMAND+=(--r4-native-mask-root "$CYJ11_R5_R4_NATIVE_MASK_ROOT")
[[ -z ${CYJ11_R5_L100_PROPOSED_CHECKPOINT:-} ]] || PLAN_COMMAND+=(--l100-proposed-checkpoint "$CYJ11_R5_L100_PROPOSED_CHECKPOINT")
if [[ -n ${CYJ11_R5_CASES:-} ]]; then
    for CASE in $CYJ11_R5_CASES; do PLAN_COMMAND+=(--case "$CASE"); done
fi

if [[ "$MODE" == "--dry-run" ]]; then
    (cd "$REPO" && "${PLAN_COMMAND[@]}")
    exit 0
fi

: "${CYJ11_R5_R2_COMPLETION_MARKER:?set CYJ11_R5_R2_COMPLETION_MARKER after the r2 import audit}"
[[ -f "$CYJ11_R5_R2_COMPLETION_MARKER" ]] || {
    echo "r2 completion marker is missing: $CYJ11_R5_R2_COMPLETION_MARKER" >&2
    exit 1
}
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet

PLAN_TMP=$(mktemp)
trap 'rm -f "$PLAN_TMP"' EXIT
(cd "$REPO" && "${PLAN_COMMAND[@]}" > "$PLAN_TMP")

RUNNER_COMMIT=$(git -C "$REPO" rev-parse HEAD)
RUN_ID="$CYJ11_R5_SUBJECT.legacy-density.$(basename "$CYJ11_R5_HISTORICAL_WORKTREE")"
[[ -z "$RUN_LABEL" ]] || RUN_ID="$RUN_ID.$RUN_LABEL"
JOBS_ROOT="$CYJ11_R5_ANALYSIS_ROOT/jobs/r5"
LOGS_ROOT="$CYJ11_R5_ANALYSIS_ROOT/logs/r5"
CONFIG_ROOT="$CYJ11_R5_ANALYSIS_ROOT/configs/r5/runners/$RUN_ID.$RUNNER_COMMIT"
ENV_ROOT="$CYJ11_R5_ANALYSIS_ROOT/environments/r5"
PLAN_PATH="$JOBS_ROOT/$RUN_ID.plan.json"
COMMAND_LOG="$LOGS_ROOT/$RUN_ID.commands.log"
[[ ! -e "$PLAN_PATH" ]] || { echo "r5 plan already exists: $PLAN_PATH" >&2; exit 1; }
[[ ! -e "$COMMAND_LOG" ]] || { echo "r5 command log already exists: $COMMAND_LOG" >&2; exit 1; }
mkdir -p "$JOBS_ROOT" "$LOGS_ROOT" "$CONFIG_ROOT" "$ENV_ROOT"
cp "$PLAN_TMP" "$PLAN_PATH"
cp "$0" "$CONFIG_ROOT/run_cyj11_r5_legacy_figures.sh"

(cd "$REPO" && "$CYJ11_R5_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git \
    --repository "$REPO" --commit "$RUNNER_COMMIT" \
    --output "$CYJ11_R5_ANALYSIS_ROOT/code_snapshots/r5/orchestrator/$RUNNER_COMMIT")
(cd "$REPO" && "$CYJ11_R5_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git \
    --repository "$CYJ11_R5_HISTORICAL_WORKTREE" --commit 00d88b45c75f272f05538997e38fbda85988e7aa \
    --output "$CYJ11_R5_ANALYSIS_ROOT/code_snapshots/r5/historical/00d88b45c75f272f05538997e38fbda85988e7aa")
(cd "$REPO" && "$CYJ11_R5_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git \
    --repository "$CYJ11_R5_DIRECT_NATIVE_WORKTREE" --commit d596308c3816864bb6e5f55a3ce91544f422fc43 \
    --output "$CYJ11_R5_ANALYSIS_ROOT/code_snapshots/r5/direct_native_compat/d596308c3816864bb6e5f55a3ce91544f422fc43")

{
    echo "python_executable=$CYJ11_R5_PYTHON"
    "$CYJ11_R5_PYTHON" --version
    "$CYJ11_R5_PYTHON" -m pip freeze
} > "$ENV_ROOT/$RUN_ID.txt"

(cd "$REPO" && "$CYJ11_R5_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r5_legacy_figures \
    execute-r5-plan --plan "$PLAN_PATH" --commands-log "$COMMAND_LOG")
