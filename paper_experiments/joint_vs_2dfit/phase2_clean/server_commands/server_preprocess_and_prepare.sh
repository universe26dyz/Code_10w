#!/usr/bin/env bash
set -euo pipefail

# Server-only raw-DICOM bridge.  The historical MATLAB entrypoint is named
# local_preprocess_all, but this launcher intentionally runs it on the server.
# The single-subject manifest must point both local_dicom_dir and
# server_dicom_dir at the same server-resident DICOM directories.
if [ "$#" -lt 4 ] || [ "$#" -gt 6 ]; then
  echo "Usage: bash $0 RUN_ROOT SUBJECT DEPLOYMENT_MANIFEST PROTOCOL [MATLAB] [PYTHON]" >&2
  exit 2
fi

RUN_ROOT=$(realpath -m "$1")
SUBJECT=$2
MANIFEST=$(realpath "$3")
PROTOCOL=$(realpath "$4")
MATLAB=${5:-matlab}
PY=${6:-python}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
PREPROCESSED_ROOT=$RUN_ROOT/preprocessed
PREPARED_ROOT=$RUN_ROOT/prepared
VALIDATION=$PREPROCESSED_ROOT/$SUBJECT/preprocess_validation.json

die() { echo "ERROR: $*" >&2; exit 2; }
require_executable() { command -v "$1" >/dev/null 2>&1 || [ -x "$1" ] || die "executable is unavailable: $1"; }

[ -d "$REPO" ] || die "repository is missing: $REPO"
[ -f "$MANIFEST" ] || die "deployment manifest is missing: $MANIFEST"
[ -f "$PROTOCOL" ] || die "protocol is missing: $PROTOCOL"
require_executable "$MATLAB"
require_executable "$PY"
[ ! -e "$PREPROCESSED_ROOT" ] || die "refusing to overwrite preprocessed output: $PREPROCESSED_ROOT"
[ ! -e "$PREPARED_ROOT" ] || die "refusing to overwrite prepared output: $PREPARED_ROOT"

"$PY" - "$MANIFEST" "$SUBJECT" <<'PY'
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
subject_id = sys.argv[2]
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if payload.get("schema_version") != "deployment-v1.1":
    raise SystemExit("deployment manifest must use schema_version deployment-v1.1")
subjects = [item for item in payload.get("subjects", []) if item.get("subject_id") == subject_id]
if len(subjects) != 1 or len(payload.get("subjects", [])) != 1:
    raise SystemExit("server-only launcher requires exactly one manifest subject matching SUBJECT")
entries = subjects[0].get("stacks", [])
if {item.get("stack") for item in entries} != {"sax", "2ch", "4ch"}:
    raise SystemExit("server-only launcher requires exactly sax, 2ch, and 4ch stacks")
for item in entries:
    local_path = Path(item["local_dicom_dir"]).resolve()
    server_path = Path(item["server_dicom_dir"]).resolve()
    if local_path != server_path:
        raise SystemExit(f"{item['stack']}: local_dicom_dir and server_dicom_dir must be the same server path")
    if not server_path.is_dir():
        raise SystemExit(f"{item['stack']}: server DICOM directory is missing: {server_path}")
    count = len(list(server_path.glob("*.dcm")))
    if count == 0 or count % 10:
        raise SystemExit(f"{item['stack']}: expected a non-zero multiple of 10 DICOM files, found {count}")
    print(f"{item['stack']}: {count} DICOM files at {server_path}")
PY

mkdir -p "$RUN_ROOT"
echo "START server-only MATLAB preprocessing: subject=$SUBJECT"
"$MATLAB" -batch "addpath('$REPO/trad/scripts'); local_preprocess_all('$MANIFEST','$PREPROCESSED_ROOT');" \
  2>&1 | tee "$RUN_ROOT/01_server_preprocess.log"

"$PY" "$REPO/trad/scripts/validate_transferred_preprocessed.py" \
  --manifest "$MANIFEST" --preprocessed-root "$PREPROCESSED_ROOT" --output "$VALIDATION" \
  2>&1 | tee "$RUN_ROOT/02_preprocessed_validation.log"

echo "START server-only observations bridge: subject=$SUBJECT"
"$PY" "$REPO/trad/scripts/prepare_all_observations.py" \
  --manifest "$MANIFEST" --preprocessed-root "$PREPROCESSED_ROOT" \
  --prepared-root "$PREPARED_ROOT" --protocol "$PROTOCOL" \
  2>&1 | tee "$RUN_ROOT/03_prepare_observations.log"

for STACK in sax 2ch 4ch; do
  [ -f "$PREPROCESSED_ROOT/$SUBJECT/$STACK/preprocessed.mat" ] || die "MAT output is missing for $STACK"
  [ -f "$PREPROCESSED_ROOT/$SUBJECT/$STACK/preprocess_qc.json" ] || die "preprocess QC is missing for $STACK"
  [ -f "$PREPARED_ROOT/$SUBJECT/$STACK/observations.npz" ] || die "observations are missing for $STACK"
  [ -f "$PREPARED_ROOT/$SUBJECT/$STACK/manifest.json" ] || die "prepared manifest is missing for $STACK"
done
echo "Server-only preprocessing and observations: PASS"
