#!/usr/bin/env bash
# Physically import completed r2 results into the unified analysis root.
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || { echo "usage: $0 [--dry-run|--execute]" >&2; exit 2; }
: "${REPO:?set REPO to the checkout containing cyj11_analysis_archive}"
: "${CYJ11_R2_PYTHON:?set CYJ11_R2_PYTHON}"
: "${CYJ11_R2_SOURCE_ROOT:?set CYJ11_R2_SOURCE_ROOT to cyj11_q002_profile_v2_20261026_r2}"
: "${CYJ11_R2_ANALYSIS_ROOT:?set CYJ11_R2_ANALYSIS_ROOT to ismrm_five_subject_analysis_v1}"
: "${CYJ11_R2_SUBJECT:?set CYJ11_R2_SUBJECT to DYZ, HJL, HHZ, or DYL_2}"

CASES=(SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH50_4CH100_fold0 S2CH50_4CH100_fold1 S2CH100_4CH50_fold0 S2CH100_4CH50_fold1)
for CASE in "${CASES[@]}"; do
    [[ -d "$CYJ11_R2_SOURCE_ROOT/$CYJ11_R2_SUBJECT/$CASE" ]] || { echo "missing r2 case: $CASE" >&2; exit 1; }
done

if [[ "$MODE" == "--dry-run" ]]; then
    printf 'r2 import subject=%s cases=%s\n' "$CYJ11_R2_SUBJECT" "${CASES[*]}"
    exit 0
fi

(cd "$REPO" && "$CYJ11_R2_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive import-r2 --r2-root "$CYJ11_R2_SOURCE_ROOT" --analysis-root "$CYJ11_R2_ANALYSIS_ROOT" --subject "$CYJ11_R2_SUBJECT")

: "${CYJ11_R2_ORCHESTRATOR_WORKTREE:?set the ed2feb7 r2 orchestration checkout}"
: "${CYJ11_R2_SPARSE_WORKTREE:?set the 6f4142 sparse worktree}"
: "${CYJ11_R2_Q002_WORKTREE:?set the de41aef Q002 worktree}"
: "${CYJ11_R2_FIT_FIRST_WORKTREE:?set the 00d88b fit-first worktree}"
: "${CYJ11_R2_RUNNER_SCRIPT:?set the exact r2 runner script path}"
R2_CODE_ROOT="$CYJ11_R2_ANALYSIS_ROOT/code_snapshots/r2"
(cd "$REPO" && "$CYJ11_R2_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R2_ORCHESTRATOR_WORKTREE" --commit ed2feb75b551db2a34adf5e94f22b54785986617 --output "$R2_CODE_ROOT/orchestrator/ed2feb75b551db2a34adf5e94f22b54785986617")
(cd "$REPO" && "$CYJ11_R2_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R2_SPARSE_WORKTREE" --commit 6f4142cccc17782ba6e985e74531822560b369f3 --output "$R2_CODE_ROOT/sparse/6f4142cccc17782ba6e985e74531822560b369f3")
(cd "$REPO" && "$CYJ11_R2_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R2_Q002_WORKTREE" --commit de41aef1c52b12037efed20f71cdf0db4696d49c --output "$R2_CODE_ROOT/q002/de41aef1c52b12037efed20f71cdf0db4696d49c")
(cd "$REPO" && "$CYJ11_R2_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive archive-git --repository "$CYJ11_R2_FIT_FIRST_WORKTREE" --commit 00d88b45c75f272f05538997e38fbda85988e7aa --output "$R2_CODE_ROOT/fit_first/00d88b45c75f272f05538997e38fbda85988e7aa")
R2_CONFIG="$CYJ11_R2_ANALYSIS_ROOT/configs/r2/$CYJ11_R2_SUBJECT"
[[ ! -e "$R2_CONFIG/run_cyj11_untrimmed_reproduction.sh" ]] || { echo "r2 config archive already exists" >&2; exit 1; }
mkdir -p "$R2_CONFIG"
cp "$CYJ11_R2_RUNNER_SCRIPT" "$R2_CONFIG/run_cyj11_untrimmed_reproduction.sh"
