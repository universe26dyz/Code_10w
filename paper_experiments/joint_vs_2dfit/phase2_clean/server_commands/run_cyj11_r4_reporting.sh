#!/usr/bin/env bash
# Immutable r4 metrics/reporting runner. Dry-run is the default.
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || {
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
}

: "${REPO:?set REPO to the checkout containing the r4 planner}"
: "${CYJ11_R4_PYTHON:?set CYJ11_R4_PYTHON}"
: "${CYJ11_R4_SUBJECT:?set CYJ11_R4_SUBJECT}"
: "${CYJ11_R4_PROTOCOL:?set CYJ11_R4_PROTOCOL to profile-v2 or legacy-density}"
: "${CYJ11_R4_ANALYSIS_ROOT:?set CYJ11_R4_ANALYSIS_ROOT}"
: "${CYJ11_R4_HISTORICAL_WORKTREE:?set CYJ11_R4_HISTORICAL_WORKTREE at 00d88b45}"
: "${CYJ11_R4_DIRECT_NATIVE_AUDIT_WORKTREE:?set CYJ11_R4_DIRECT_NATIVE_AUDIT_WORKTREE at 5ceebbe}"

case "$CYJ11_R4_SUBJECT" in CYJ|DYZ|HJL|HHZ|DYL_2) ;; *) echo "invalid CYJ11_R4_SUBJECT: $CYJ11_R4_SUBJECT" >&2; exit 2;; esac
case "$CYJ11_R4_PROTOCOL" in profile-v2|legacy-density) ;; *) echo "invalid CYJ11_R4_PROTOCOL: $CYJ11_R4_PROTOCOL" >&2; exit 2;; esac
case "${CYJ11_R4_RENDER_FIGURES:-0}" in 0|1) ;; *) echo "CYJ11_R4_RENDER_FIGURES must be 0 or 1" >&2; exit 2;; esac
RUN_LABEL=${CYJ11_R4_RUN_LABEL:-}
[[ -z "$RUN_LABEL" || "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "CYJ11_R4_RUN_LABEL must contain only letters, digits, dot, underscore, or dash" >&2
    exit 2
}

PLAN_COMMAND=(
    "$CYJ11_R4_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting
    --subject "$CYJ11_R4_SUBJECT" --protocol "$CYJ11_R4_PROTOCOL"
    --analysis-root "$CYJ11_R4_ANALYSIS_ROOT" --historical-worktree "$CYJ11_R4_HISTORICAL_WORKTREE"
    --direct-native-audit-worktree "$CYJ11_R4_DIRECT_NATIVE_AUDIT_WORKTREE"
    --python-executable "$CYJ11_R4_PYTHON" --device "${CYJ11_R4_DEVICE:-cuda:0}"
)
[[ -z ${CYJ11_R4_PREPARED_ROOT:-} ]] || PLAN_COMMAND+=(--prepared-root "$CYJ11_R4_PREPARED_ROOT")
[[ -z ${CYJ11_R4_R0_NATIVE_MAPS_ROOT:-} ]] || PLAN_COMMAND+=(--r0-native-maps-root "$CYJ11_R4_R0_NATIVE_MAPS_ROOT")
[[ -z ${CYJ11_R4_L100_PROPOSED_CHECKPOINT:-} ]] || PLAN_COMMAND+=(--l100-proposed-checkpoint "$CYJ11_R4_L100_PROPOSED_CHECKPOINT")
if [[ -n ${CYJ11_R4_CASES:-} ]]; then
    for CASE in $CYJ11_R4_CASES; do PLAN_COMMAND+=(--case "$CASE"); done
fi
[[ ${CYJ11_R4_RENDER_FIGURES:-0} == 1 ]] && PLAN_COMMAND+=(--render-figures)

if [[ "$MODE" == "--dry-run" ]]; then
    (cd "$REPO" && "${PLAN_COMMAND[@]}")
    exit 0
fi

: "${CYJ11_R4_R2_COMPLETION_MARKER:?set CYJ11_R4_R2_COMPLETION_MARKER after the r2 import audit}"
[[ -f "$CYJ11_R4_R2_COMPLETION_MARKER" ]] || { echo "r2 completion marker is missing: $CYJ11_R4_R2_COMPLETION_MARKER" >&2; exit 1; }
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet

PLAN_TMP=$(mktemp)
trap 'rm -f "$PLAN_TMP"' EXIT
(cd "$REPO" && "${PLAN_COMMAND[@]}") > "$PLAN_TMP"

JOBS_ROOT="$CYJ11_R4_ANALYSIS_ROOT/jobs/r4"
LOGS_ROOT="$CYJ11_R4_ANALYSIS_ROOT/logs/r4"
RUNNER_COMMIT=$(git -C "$REPO" rev-parse HEAD)
RUN_ID="$CYJ11_R4_SUBJECT.$CYJ11_R4_PROTOCOL.$(basename "$CYJ11_R4_HISTORICAL_WORKTREE")"
[[ -z "$RUN_LABEL" ]] || RUN_ID="$RUN_ID.$RUN_LABEL"
CONFIG_ROOT="$CYJ11_R4_ANALYSIS_ROOT/configs/r4/runners/$RUN_ID.$RUNNER_COMMIT"
ENV_ROOT="$CYJ11_R4_ANALYSIS_ROOT/environments/r4"
PLAN_PATH="$JOBS_ROOT/$RUN_ID.plan.json"
COMMAND_LOG="$LOGS_ROOT/$RUN_ID.commands.log"
[[ ! -e "$PLAN_PATH" ]] || { echo "r4 plan already exists: $PLAN_PATH" >&2; exit 1; }
[[ ! -e "$COMMAND_LOG" ]] || { echo "r4 command log already exists: $COMMAND_LOG" >&2; exit 1; }
mkdir -p "$JOBS_ROOT" "$LOGS_ROOT" "$CONFIG_ROOT" "$ENV_ROOT"
cp "$PLAN_TMP" "$PLAN_PATH"
cp "$0" "$CONFIG_ROOT/run_cyj11_r4_reporting.sh"
printf '%q ' "$0" --execute > "$PLAN_PATH.command.sh"
printf '\n' >> "$PLAN_PATH.command.sh"

(cd "$REPO" && "$CYJ11_R4_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$REPO" --commit "$RUNNER_COMMIT" --output "$CYJ11_R4_ANALYSIS_ROOT/code_snapshots/r4/orchestrator/$RUNNER_COMMIT")
(cd "$REPO" && "$CYJ11_R4_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R4_HISTORICAL_WORKTREE" --commit 00d88b45c75f272f05538997e38fbda85988e7aa --output "$CYJ11_R4_ANALYSIS_ROOT/code_snapshots/r4/historical/00d88b45c75f272f05538997e38fbda85988e7aa")
(cd "$REPO" && "$CYJ11_R4_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R4_DIRECT_NATIVE_AUDIT_WORKTREE" --commit 5ceebbeb5f510cb4cf53b3d2e0744d3ab62e3d0a --output "$CYJ11_R4_ANALYSIS_ROOT/code_snapshots/r4/direct_native_audit/5ceebbeb5f510cb4cf53b3d2e0744d3ab62e3d0a")
{
    echo "python_executable=$CYJ11_R4_PYTHON"
    "$CYJ11_R4_PYTHON" --version
    "$CYJ11_R4_PYTHON" -m pip freeze
} > "$ENV_ROOT/$RUN_ID.txt"

(cd "$REPO" && "$CYJ11_R4_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting execute-r4-plan --plan "$PLAN_PATH" --commands-log "$COMMAND_LOG")
