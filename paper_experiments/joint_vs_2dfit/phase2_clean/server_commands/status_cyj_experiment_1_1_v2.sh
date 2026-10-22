#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "status is read-only" >&2; exit 2; }
manifest=${2:?usage: $0 --dry-run SPARSE_MANIFEST}
[[ $# -eq 2 ]] || { echo "unexpected arguments" >&2; exit 2; }
python=${PYTHON:-python3}
"$python" - "$manifest" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); data = json.loads(path.read_text(encoding="utf-8"))
if data.get("schema_version") != "phase4r_sparse_profile_v2": raise SystemExit("invalid sparse profile manifest schema")
if not all(isinstance(data.get(key), str) and data[key] for key in ("subject", "profile_id", "case_id")): raise SystemExit("missing subject/profile_id/case_id")
for stack in ("sax", "2ch", "4ch"):
    item = data.get("stacks", {}).get(stack)
    if not isinstance(item, dict) or not isinstance(item.get("training_stack_present"), bool): raise SystemExit(f"{stack}: missing training_stack_present")
print(f"read-only profile status: subject={data['subject']} profile_id={data['profile_id']} case_id={data['case_id']}")
print(f"future output root (not created): {path.parent.parent / 'formal_runs' / data['case_id']}")
print("future command (not executed): python -m quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction --sparse-manifest", path)
PY
