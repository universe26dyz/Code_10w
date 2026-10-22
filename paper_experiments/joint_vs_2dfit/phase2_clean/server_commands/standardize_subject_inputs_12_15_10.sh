#!/usr/bin/env bash
set -euo pipefail
mode=${1:---dry-run}
[[ "$mode" == "--dry-run" ]] || { echo "standardization execution is not authorized by this wrapper" >&2; exit 2; }
shift || true
subject=${1:?usage: $0 --dry-run SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT}
prepared_subject_root=${2:?usage: $0 --dry-run SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT}
output_root=${3:?usage: $0 --dry-run SUBJECT PREPARED_SUBJECT_ROOT OUTPUT_ROOT}
[[ $# -eq 3 ]] || { echo "unexpected arguments" >&2; exit 2; }
python=${PYTHON:-python}
cmd=("$python" -m paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices --prepared-subject-root "$prepared_subject_root" --subject "$subject" --output-root "$output_root" --dry-run)
printf 'dry-run only; subject=%s\nsource_root=%s\noutput_root=%s\ntargets: SAX=12 2CH=15 4CH=10\n' "$subject" "$prepared_subject_root" "$output_root"
printf 'resolved trim mapping and geometry audit (read-only):\n'
"${cmd[@]}"
printf 'future immutable command (not executed): '
printf '%q ' "${cmd[@]:0:${#cmd[@]}-1}"
printf '\n'
