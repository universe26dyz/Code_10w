#!/usr/bin/env bash
# Independent CYJ1.1 legacy-density r3 runner.  Dry-run is default.
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || {
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
}

: "${REPO:?set REPO to the checkout containing the r3 planner}"
: "${CYJ11_R3_PYTHON:?set CYJ11_R3_PYTHON}"
: "${CYJ11_R3_SUBJECT:?set CYJ11_R3_SUBJECT to DYZ, HJL, HHZ, or DYL_2}"
: "${CYJ11_R3_SOURCE_PREPARED_ROOT:?set CYJ11_R3_SOURCE_PREPARED_ROOT}"
: "${CYJ11_R3_ANALYSIS_ROOT:?set CYJ11_R3_ANALYSIS_ROOT}"
: "${CYJ11_R3_HISTORICAL_WORKTREE:?set CYJ11_R3_HISTORICAL_WORKTREE at dad9698b}"
: "${CYJ11_R3_B6_MODEL:?set CYJ11_R3_B6_MODEL}"
: "${CYJ11_R3_SIGNAL_SIMULATOR:?set CYJ11_R3_SIGNAL_SIMULATOR}"
: "${CYJ11_R3_NESVOR_EXECUTABLE:?set CYJ11_R3_NESVOR_EXECUTABLE}"
: "${CYJ11_R3_EXPECTED_B6_SHA256:?set CYJ11_R3_EXPECTED_B6_SHA256}"
: "${CYJ11_R3_EXPECTED_SIGNAL_SIMULATOR_SHA256:?set CYJ11_R3_EXPECTED_SIGNAL_SIMULATOR_SHA256}"

case "$CYJ11_R3_SUBJECT" in DYZ|HJL|HHZ|DYL_2) ;; *) echo "invalid CYJ11_R3_SUBJECT: $CYJ11_R3_SUBJECT" >&2; exit 2;; esac

PLAN_COMMAND=(
    "$CYJ11_R3_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction
    --subject "$CYJ11_R3_SUBJECT" --source-prepared-root "$CYJ11_R3_SOURCE_PREPARED_ROOT"
    --analysis-root "$CYJ11_R3_ANALYSIS_ROOT" --historical-worktree "$CYJ11_R3_HISTORICAL_WORKTREE"
    --b6-model "$CYJ11_R3_B6_MODEL" --signal-simulator "$CYJ11_R3_SIGNAL_SIMULATOR"
    --nesvor-executable "$CYJ11_R3_NESVOR_EXECUTABLE"
    --expected-b6-model-sha256 "$CYJ11_R3_EXPECTED_B6_SHA256"
    --expected-signal-simulator-sha256 "$CYJ11_R3_EXPECTED_SIGNAL_SIMULATOR_SHA256"
    --python-executable "$CYJ11_R3_PYTHON" --device "${CYJ11_R3_DEVICE:-cuda:0}"
)

if [[ "$MODE" == "--dry-run" ]]; then
    (cd "$REPO" && "${PLAN_COMMAND[@]}")
    exit 0
fi

: "${CYJ11_R3_R2_COMPLETION_MARKER:?set CYJ11_R3_R2_COMPLETION_MARKER only after r2 is complete and audited}"
[[ -f "$CYJ11_R3_R2_COMPLETION_MARKER" ]] || { echo "r2 completion marker is missing: $CYJ11_R3_R2_COMPLETION_MARKER" >&2; exit 1; }
git -C "$REPO" diff --quiet
git -C "$REPO" diff --cached --quiet

PLAN_TMP=$(mktemp)
trap 'rm -f "$PLAN_TMP"' EXIT
(cd "$REPO" && "${PLAN_COMMAND[@]}") > "$PLAN_TMP"

JOBS_ROOT="$CYJ11_R3_ANALYSIS_ROOT/jobs/r3"
LOGS_ROOT="$CYJ11_R3_ANALYSIS_ROOT/logs/r3"
CONFIG_ROOT="$CYJ11_R3_ANALYSIS_ROOT/configs/r3/$CYJ11_R3_SUBJECT"
ENV_ROOT="$CYJ11_R3_ANALYSIS_ROOT/environments/r3"
[[ ! -e "$JOBS_ROOT/$CYJ11_R3_SUBJECT.plan.json" ]] || { echo "r3 plan already exists" >&2; exit 1; }
[[ ! -e "$LOGS_ROOT/$CYJ11_R3_SUBJECT.commands.log" ]] || { echo "r3 command log already exists" >&2; exit 1; }
mkdir -p "$JOBS_ROOT" "$LOGS_ROOT" "$CONFIG_ROOT" "$ENV_ROOT"
cp "$PLAN_TMP" "$JOBS_ROOT/$CYJ11_R3_SUBJECT.plan.json"
cp "$0" "$CONFIG_ROOT/run_cyj11_r3_reproduction.sh"
printf '%q ' "$0" --execute > "$JOBS_ROOT/$CYJ11_R3_SUBJECT.command.sh"
printf '\n' >> "$JOBS_ROOT/$CYJ11_R3_SUBJECT.command.sh"

RUNNER_COMMIT=$(git -C "$REPO" rev-parse HEAD)
(cd "$REPO" && "$CYJ11_R3_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$REPO" --commit "$RUNNER_COMMIT" --output "$CYJ11_R3_ANALYSIS_ROOT/code_snapshots/r3/orchestrator/$RUNNER_COMMIT")
(cd "$REPO" && "$CYJ11_R3_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R3_HISTORICAL_WORKTREE" --commit dad9698b5b2b5cd98267a6d23f1eb866b54791d3 --output "$CYJ11_R3_ANALYSIS_ROOT/code_snapshots/r3/historical/dad9698b5b2b5cd98267a6d23f1eb866b54791d3")
{
    echo "python_executable=$CYJ11_R3_PYTHON"
    "$CYJ11_R3_PYTHON" --version
    echo "nesvor_executable=$CYJ11_R3_NESVOR_EXECUTABLE"
    sha256sum "$CYJ11_R3_NESVOR_EXECUTABLE"
    "$CYJ11_R3_PYTHON" -m pip freeze
} > "$ENV_ROOT/$CYJ11_R3_SUBJECT.txt"

(cd "$REPO" && "$CYJ11_R3_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive execute-r3-plan --plan "$JOBS_ROOT/$CYJ11_R3_SUBJECT.plan.json" --commands-log "$LOGS_ROOT/$CYJ11_R3_SUBJECT.commands.log")
