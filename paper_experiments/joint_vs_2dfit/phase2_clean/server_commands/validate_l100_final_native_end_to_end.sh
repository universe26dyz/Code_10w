#!/usr/bin/env bash
# Server-only L100 consumer-side validation. It never trains or reconstructs.
set -euo pipefail
[[ $# -eq 10 ]] || { echo "usage: $0 FIT_POSE_AUDIT_V3 FIT_FIRST_BUNDLE DIRECT_BUNDLE NATIVE_MASK_ROOT R0_NATIVE_ROOT L100_SPARSE_MANIFEST FIGURE_CONFIG_JSON NEW_METRICS_ROOT NEW_FIGURE_ROOT PYTHON" >&2; exit 2; }
AUDIT=$(realpath -m "$1"); FIT=$(realpath -m "$2"); DIRECT=$(realpath -m "$3"); MASKS=$(realpath -m "$4"); R0=$(realpath -m "$5"); SPARSE=$(realpath -m "$6"); CONFIG=$(realpath -m "$7"); METRICS=$(realpath -m "$8"); FIGURES=$(realpath -m "$9"); PY=${10}
for path in "$AUDIT/manifest.json" "$FIT/manifest.json" "$DIRECT/manifest.json" "$MASKS/manifest.json" "$SPARSE" "$CONFIG"; do test -f "$path" || { echo "missing prerequisite: $path" >&2; exit 1; }; done
test -d "$R0" || { echo "missing prerequisite: $R0" >&2; exit 1; }
test ! -e "$METRICS" || { echo "refusing existing metrics output: $METRICS" >&2; exit 1; }
test ! -e "$FIGURES" || { echo "refusing existing figure output: $FIGURES" >&2; exit 1; }
"$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); assert a.get("schema")=="fit_first_native_pose_audit/v3" and a.get("reconstruction_input_geometry_gate",{}).get("status")=="RECONSTRUCTION_INPUT_GEOMETRY_VALID", "Fit-first v3 geometry gate is not valid"; assert c.get("condition")=="L100_fold0" and c.get("native_mask_root"), "config must be L100_fold0 with native_mask_root"; assert c.get("scmr_longitudinal") is not None, "config must request SCMR-style longitudinal figures"' "$AUDIT/manifest.json" "$CONFIG"
REPO=${REPO:-/data/dengyz/code/Code_10w}; mkdir -p "$(dirname "$METRICS")" "$(dirname "$FIGURES")"; cd "$REPO"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final_pose_native_metrics --direct-bundle-root "$DIRECT" --fit-first-bundle-root "$FIT" --r0-native-maps-root "$R0" --sparse-manifest "$SPARSE" --mask-root "$MASKS" --output "$METRICS"
"$PY" -m paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting figures --config "$CONFIG" --output "$FIGURES"
