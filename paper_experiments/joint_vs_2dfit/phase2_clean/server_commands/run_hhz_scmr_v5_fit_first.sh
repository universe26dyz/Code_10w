#!/usr/bin/env bash
set -euo pipefail

# Rebuild HHZ native maps with the retained SCMR-v5 MATLAB matcher, then use
# that immutable cache for L100 and sparse Fit-first runs.  This script never
# accepts the generic Python/TRAD matcher as an HHZ prebuilt-map substitute.
#
# Usage:
#   bash run_hhz_scmr_v5_fit_first.sh RUN_ROOT PREPROCESSED_ROOT [PREPARED_ROOT] [MATLAB]

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash $0 RUN_ROOT PREPROCESSED_ROOT [PREPARED_ROOT] [MATLAB]" >&2
  exit 2
fi

RUN_ROOT=$(realpath "$1")
PREPROCESSED_ROOT=$(realpath "$2")
PREPARED_ROOT=$(realpath "${3:-/data/dengyz/dataset/Code_10w_v1/Code_10w_prepared_scmr_v5}")
MATLAB=${4:-matlab}
REPO=${REPO:-/data/dengyz/code/Code_10w}
PY=${PY:-/home/dengyz/miniconda3/envs/cr_dreme/bin/python}
NESVOR=${NESVOR:-/home/dengyz/miniconda3/envs/nesvor/bin/nesvor}
SUBJECT=${SUBJECT:-HHZ}
CACHE=$RUN_ROOT/native_cache/HHZ_scmr_v5
MAPPING_ROOT=$RUN_ROOT/mapping/scmr_v5/$SUBJECT

die() { echo "ERROR: $*" >&2; exit 2; }
require_executable() { [ -x "$1" ] || die "executable is missing or not executable: $1"; }
require_directory() { [ -d "$1" ] || die "directory is missing: $1"; }
require_absent() { [ ! -e "$1" ] || die "refusing to overwrite existing path: $1"; }

require_executable "$PY"
require_executable "$NESVOR"
require_directory "$REPO"
require_directory "$PREPROCESSED_ROOT/$SUBJECT"
require_directory "$PREPARED_ROOT/$SUBJECT"
require_absent "$CACHE"
require_absent "$MAPPING_ROOT"
echo "SCMR-v5 Fit-first inputs:"
echo "  RUN_ROOT=$RUN_ROOT"
echo "  PREPROCESSED_ROOT=$PREPROCESSED_ROOT"
echo "  PREPARED_ROOT=$PREPARED_ROOT"
echo "  PY=$PY"
echo "  NESVOR=$NESVOR"
echo "  CACHE=$CACHE"
echo "  MAPPING_ROOT=$MAPPING_ROOT"
mkdir -p "$RUN_ROOT/mapping/scmr_v5"
cd "$REPO"

"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.scmr_v5_native_cache \
  --subject "$SUBJECT" --prepared-root "$PREPARED_ROOT" \
  --preprocessed-root "$PREPROCESSED_ROOT" --mapping-root "$MAPPING_ROOT" \
  --output "$CACHE" --matlab-executable "$MATLAB" 2>&1 | tee "$RUN_ROOT/hhz_scmr_v5_native_cache.log"

L100_OUTPUT=$RUN_ROOT/reconstruction/fit_first/$SUBJECT/L100_fold0
test ! -e "$L100_OUTPUT"
mkdir -p "$(dirname "$L100_OUTPUT")"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline \
  --prepared-root "$PREPARED_ROOT" --subject-id "$SUBJECT" \
  --prebuilt-native-maps-root "$CACHE" --require-scmr-v5-native-maps \
  --nesvor-executable "$NESVOR" --output "$L100_OUTPUT" --execute 2>&1 | tee "$L100_OUTPUT.log"

for CASE_ROOT in "$RUN_ROOT"/sparse_inputs/$SUBJECT/*; do
  [ -d "$CASE_ROOT" ] || continue
  CASE=$(basename "$CASE_ROOT")
  [ "$CASE" = "L100_fold0" ] && continue
  MANIFEST=$CASE_ROOT/sparse_manifest.json
  OUTPUT=$RUN_ROOT/reconstruction/fit_first/$SUBJECT/$CASE
  test -f "$MANIFEST"
  test ! -e "$OUTPUT"
  read -r GENERATOR_SHA MANIFEST_SHA < <("$PY" - "$MANIFEST" <<'PY'
import hashlib, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
print(payload["git_sha"], hashlib.sha256(path.read_bytes()).hexdigest())
PY
  )
  mkdir -p "$(dirname "$OUTPUT")"
  "$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline \
    --prepared-root "$PREPARED_ROOT" --subject-id "$SUBJECT" \
    --reuse-native-maps-root "$CACHE" --sparse-manifest "$MANIFEST" \
    --expected-generator-git-sha "$GENERATOR_SHA" \
    --expected-manifest-sha256 "$MANIFEST_SHA" \
    --nesvor-executable "$NESVOR" --output "$OUTPUT" --execute 2>&1 | tee "$OUTPUT.log"
done

echo "SCMR-v5 Fit-first native maps and reconstructions: PASS"
