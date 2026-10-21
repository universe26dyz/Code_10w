#!/usr/bin/env bash
# Server-only held-out L50/L25 consumer validation; never trains or reconstructs.
set -euo pipefail
[[ $# -eq 10 ]] || { echo "usage: $0 PROPOSED_FORMAL_ENDPOINT_ROOT FIT_FIRST_FORMAL_ENDPOINT_ROOT NATIVE_MASK_ROOT R0_NATIVE_ROOT SPARSE_MANIFEST FIGURE_CONFIG_JSON NEW_METRICS_ROOT NEW_FIGURE_ROOT PYTHON REPO" >&2; exit 2; }
PROPOSED=$(realpath -m "$1"); FIT=$(realpath -m "$2"); MASKS=$(realpath -m "$3"); R0=$(realpath -m "$4"); SPARSE=$(realpath -m "$5"); CONFIG=$(realpath -m "$6"); METRICS=$(realpath -m "$7"); FIGURES=$(realpath -m "$8"); PY=$9; REPO=${10}
for path in "$PROPOSED/manifest.json" "$FIT/manifest.json" "$MASKS/manifest.json" "$SPARSE" "$CONFIG"; do test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }; done
test -d "$R0"; test ! -e "$METRICS"; test ! -e "$FIGURES"
"$PY" -c 'import json,sys; s=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); assert s.get("condition", "").startswith(("L50_fold", "L25_fold")), "sparse wrapper requires L50/L25"; assert c.get("condition")==s["condition"] and c.get("native_mask_root"), "figure config condition/native mask mismatch"' "$SPARSE" "$CONFIG"
mkdir -p "$(dirname "$METRICS")" "$(dirname "$FIGURES")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final_sparse_native_metrics --proposed-endpoint-root "$PROPOSED" --fit-first-endpoint-root "$FIT" --native-mask-root "$MASKS" --r0-native-maps-root "$R0" --sparse-manifest "$SPARSE" --output "$METRICS"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting figures --config "$CONFIG" --output "$FIGURES"
