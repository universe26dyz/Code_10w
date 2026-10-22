#!/usr/bin/env bash
set -euo pipefail
mode=${1:---dry-run}
[[ "$mode" == "--dry-run" ]] || { echo "audit is read-only; execution mode is rejected" >&2; exit 2; }
shift || true
standardization_manifest=${1:?usage: $0 --dry-run STANDARDIZATION_MANIFEST [REUSE_MANIFEST]}
reuse_manifest=${2:-}
[[ $# -le 2 ]] || { echo "unexpected arguments" >&2; exit 2; }
python=${PYTHON:-python}
"$python" - "$standardization_manifest" "$reuse_manifest" <<'PY'
import hashlib, json, sys
from pathlib import Path
standardization = Path(sys.argv[1]); reuse = Path(sys.argv[2]) if sys.argv[2] else None
payload = json.loads(standardization.read_text(encoding="utf-8"))
if payload.get("schema") != "phase4r_common_slice_standardization/v1": raise SystemExit("invalid standardization schema")
for stack, target in {"sax": 12, "2ch": 15, "4ch": 10}.items():
    item = payload.get("stacks", {}).get(stack, {})
    path = Path(item.get("output_observation", {}).get("path", ""))
    if item.get("output_count") != target or not path.is_file(): raise SystemExit(f"{stack}: missing or wrong standardized output")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("output_observation", {}).get("sha256"): raise SystemExit(f"{stack}: output SHA256 mismatch")
if reuse:
    audit = json.loads(reuse.read_text(encoding="utf-8"))
    if audit.get("status") != "REUSE_VERIFIED": raise SystemExit("dictionary reuse is not REUSE_VERIFIED")
print("read-only audit: standardization outputs and declared reuse status validated")
PY
