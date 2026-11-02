#!/usr/bin/env bash
set -euo pipefail

# Build all standard sparse cases from one server-resident prepared root.
# The profiles carry no subject-specific paths; group selection is derived from
# each subject's prepared observations and recorded in immutable manifests.
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
  echo "Usage: bash $0 PREPARED_ROOT SUBJECT OUTPUT_ROOT [PYTHON] [PROFILE_ROOT]" >&2
  exit 2
fi

PREPARED_ROOT=$(realpath "$1")
SUBJECT=$2
OUTPUT_ROOT=$(realpath -m "$3")
PY=${4:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
PROFILE_ROOT=${5:-$REPO/paper_experiments/joint_vs_2dfit/configs/sparse_profiles}

die() { echo "ERROR: $*" >&2; exit 2; }
require_executable() { command -v "$1" >/dev/null 2>&1 || [ -x "$1" ] || die "executable is unavailable: $1"; }

[ -d "$REPO" ] || die "repository is missing: $REPO"
[ -d "$PREPARED_ROOT/$SUBJECT" ] || die "prepared subject is missing: $PREPARED_ROOT/$SUBJECT"
require_executable "$PY"
[ ! -e "$OUTPUT_ROOT/$SUBJECT" ] || die "refusing to overwrite sparse inputs: $OUTPUT_ROOT/$SUBJECT"
for PROFILE in SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH100_4CH50 S2CH50_4CH100; do
  [ -f "$PROFILE_ROOT/$PROFILE.json" ] || die "generic sparse profile is missing: $PROFILE_ROOT/$PROFILE.json"
done

mkdir -p "$OUTPUT_ROOT"
cd "$REPO"
for CONDITION in L100 L25 L50; do
  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse \
    --prepared-root "$PREPARED_ROOT" --subject "$SUBJECT" \
    --condition "$CONDITION" --phase 0 --output-root "$OUTPUT_ROOT"
done
for PROFILE in SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH100_4CH50 S2CH50_4CH100; do
  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.sparse \
    --prepared-root "$PREPARED_ROOT" --subject "$SUBJECT" \
    --profile "$PROFILE_ROOT/$PROFILE.json" --fold 0 --output-root "$OUTPUT_ROOT"
done

for CASE in L100_fold0 L25_fold0 L50_fold0 SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH100_4CH50_fold0 S2CH50_4CH100_fold0; do
  [ -f "$OUTPUT_ROOT/$SUBJECT/$CASE/sparse_manifest.json" ] || die "sparse manifest is missing: $CASE"
done
echo "Server sparse profiles: PASS"
