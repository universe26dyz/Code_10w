#!/usr/bin/env bash
set -euo pipefail
[[ ${1:---dry-run} == "--dry-run" ]] || { echo "status is read-only" >&2; exit 2; }
manifest=${2:?usage: $0 --dry-run Q005_RUN_MANIFEST}
[[ $# -eq 2 ]] || { echo "unexpected arguments" >&2; exit 2; }
python=${PYTHON:-python3}
"$python" - "$manifest" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); data = json.loads(path.read_text(encoding="utf-8"))
if data.get("experiment_id") != "Q005S640_K4_B6_roi_same_anchor_mse": raise SystemExit("not a Q005 K4 manifest")
for key, expected in (("training_psf_samples", 4), ("data_psf_inr_location_count", 2560), ("psf_forward_chunk_size", 4)):
    if data.get(key) != expected: raise SystemExit(f"Q005 control mismatch: {key}")
endpoint = data.get("endpoint_semantics", {})
if not isinstance(endpoint, dict) or not {"central", "psf128"}.issubset(endpoint): raise SystemExit("missing central/psf128 endpoint semantics")
print("read-only Q005 status: K=4 locations=2560 chunk=4; central/psf128 declared")
print(f"future output root (not created): {path.parent / 'endpoints'}")
print("future command (not executed): python -m quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction --experiment-id Q005S640_K4_B6_roi_same_anchor_mse ...")
PY
