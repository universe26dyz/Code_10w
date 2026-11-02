#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${1:?usage: $0 RUN_ROOT SUBJECT CASE [PYTHON]}
SUBJECT=${2:?usage: $0 RUN_ROOT SUBJECT CASE [PYTHON]}
CASE=${3:?usage: $0 RUN_ROOT SUBJECT CASE [PYTHON]}
PYTHON=${4:-python}
SPARSE="$RUN_ROOT/sparse_inputs/$SUBJECT/$CASE"
MANIFEST="$SPARSE/sparse_manifest.json"
OUT="$RUN_ROOT/reconstruction/proposed/$SUBJECT/$CASE"
[ -f "$MANIFEST" ] || { echo "missing sparse manifest: $MANIFEST" >&2; exit 1; }
[ ! -e "$OUT" ] || { echo "refusing non-fresh output: $OUT" >&2; exit 1; }
if [ "$CASE" = L100_fold0 ]; then INPUT="$RUN_ROOT/prepared/$SUBJECT"; else INPUT="$SPARSE"; fi
GENERATOR_SHA=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["git_sha"])' "$MANIFEST")
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
DECODER="$PROJECT_ROOT/runtime/model_assets/decoder_scmr_v5_10_100.pth"
B6="$PROJECT_ROOT/runtime/model_assets/b6_model.pt"
PROTOCOL="$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml"
mkdir -p "$(dirname "$OUT")"
PYTHONPATH="$PROJECT_ROOT/common${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.proposed \
  --prepared-root "$INPUT" --subject-id "$SUBJECT" --decoder "$DECODER" --b6-model "$B6" \
  --protocol "$PROTOCOL" --sparse-manifest "$MANIFEST" \
  --expected-sparse-generator-git-sha "$GENERATOR_SHA" --expected-sparse-manifest-sha256 "$MANIFEST_SHA" \
  --expected-b6-model-sha256 15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5 \
  --expected-decoder-sha256 338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b \
  --expected-protocol-sha256 3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7 \
  --output "$OUT" --device cuda:0 --execute

