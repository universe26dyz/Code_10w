#!/usr/bin/env bash
# Strict CYJ 1.1 reproduction.  Dry-run is read-only and is the default.
set -euo pipefail

MODE=${1:---dry-run}
[[ "$MODE" == "--dry-run" || "$MODE" == "--execute" ]] || {
    echo "usage: $0 [--dry-run|--execute]" >&2
    exit 2
}

: "${REPO:?set REPO to the current Code_10w checkout containing the planner}"
: "${CYJ11_PYTHON:?set CYJ11_PYTHON}"
: "${CYJ11_PREPARED_ROOT:?set CYJ11_PREPARED_ROOT to untrimmed Code_10w_prepared}"
: "${CYJ11_B6_MODEL:?set CYJ11_B6_MODEL}"
: "${CYJ11_SIGNAL_SIMULATOR:?set CYJ11_SIGNAL_SIMULATOR}"
: "${CYJ11_PROTOCOL:?set CYJ11_PROTOCOL}"
: "${CYJ11_NATIVE_CACHE_ROOT:?set CYJ11_NATIVE_CACHE_ROOT}"
: "${CYJ11_NATIVE_MASK_ROOT:?set CYJ11_NATIVE_MASK_ROOT}"
: "${CYJ11_L100_FIT_FIRST_ROOT:?set CYJ11_L100_FIT_FIRST_ROOT}"
: "${CYJ11_OUTPUT_ROOT:?set CYJ11_OUTPUT_ROOT to a new experiment root}"
: "${CYJ11_SPARSE_WORKTREE:?set CYJ11_SPARSE_WORKTREE at 6f4142...}"
: "${CYJ11_Q002_WORKTREE:?set CYJ11_Q002_WORKTREE at de41aef...}"
: "${CYJ11_FIT_FIRST_WORKTREE:?set CYJ11_FIT_FIRST_WORKTREE at 00d88b...}"
: "${CYJ11_EXPECTED_B6_SHA256:?set CYJ11_EXPECTED_B6_SHA256}"
: "${CYJ11_EXPECTED_SIGNAL_SIMULATOR_SHA256:?set CYJ11_EXPECTED_SIGNAL_SIMULATOR_SHA256}"
: "${CYJ11_EXPECTED_PROTOCOL_SHA256:?set CYJ11_EXPECTED_PROTOCOL_SHA256}"
: "${CYJ11_NESVOR_EXECUTABLE:?set CYJ11_NESVOR_EXECUTABLE}"

SUBJECT=${CYJ11_SUBJECT:-CYJ}
CASE_ID=${CYJ11_CASE:-SAX100_2CH100_4CH0}
PLAN_COMMAND=(
    "$CYJ11_PYTHON" -m paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction
    --subject "$SUBJECT" --case "$CASE_ID"
    --prepared-root "$CYJ11_PREPARED_ROOT" --output-root "$CYJ11_OUTPUT_ROOT"
    --sparse-worktree "$CYJ11_SPARSE_WORKTREE" --q002-worktree "$CYJ11_Q002_WORKTREE" --fit-first-worktree "$CYJ11_FIT_FIRST_WORKTREE"
    --b6-model "$CYJ11_B6_MODEL" --signal-simulator "$CYJ11_SIGNAL_SIMULATOR" --protocol "$CYJ11_PROTOCOL"
    --native-cache-root "$CYJ11_NATIVE_CACHE_ROOT" --native-mask-root "$CYJ11_NATIVE_MASK_ROOT"
    --l100-fit-first-root "$CYJ11_L100_FIT_FIRST_ROOT"
    --expected-b6-model-sha256 "$CYJ11_EXPECTED_B6_SHA256" --expected-signal-simulator-sha256 "$CYJ11_EXPECTED_SIGNAL_SIMULATOR_SHA256" --expected-protocol-sha256 "$CYJ11_EXPECTED_PROTOCOL_SHA256"
    --nesvor-executable "$CYJ11_NESVOR_EXECUTABLE" --python-executable "$CYJ11_PYTHON" --device "${CYJ11_DEVICE:-cuda:0}"
)

if [[ "$MODE" == "--dry-run" ]]; then
    ( cd "$REPO" && "${PLAN_COMMAND[@]}" )
    exit 0
fi

if [[ -e "$CYJ11_OUTPUT_ROOT" && ( ! -d "$CYJ11_OUTPUT_ROOT" || -n "$(find "$CYJ11_OUTPUT_ROOT" -mindepth 1 -print -quit)" ) ]]; then
    echo "refusing non-empty output root: $CYJ11_OUTPUT_ROOT" >&2
    exit 1
fi

PLAN_TMP=$(mktemp)
trap 'rm -f "$PLAN_TMP"' EXIT
( cd "$REPO" && "${PLAN_COMMAND[@]}" ) > "$PLAN_TMP"
mkdir -p "$CYJ11_OUTPUT_ROOT"
cp "$PLAN_TMP" "$CYJ11_OUTPUT_ROOT/reproduction_plan.json"

"$CYJ11_PYTHON" - "$PLAN_TMP" "$CYJ11_OUTPUT_ROOT/commands.log" <<'PY'
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

plan_path, log_path = map(Path, sys.argv[1:])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
manifest = Path(plan["runtime_bindings"]["sparse_manifest"])
placeholder = plan["runtime_bindings"]["sparse_manifest_sha256"]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def resolve(value: str, manifest_sha256: str | None) -> str:
    if value == placeholder:
        if manifest_sha256 is None:
            raise RuntimeError("sparse manifest SHA256 requested before sparse input was created")
        return manifest_sha256
    return value

manifest_sha256 = None
with log_path.open("x", encoding="utf-8") as log:
    for step in plan["steps"]:
        argv = step.get("argv")
        if argv is None:
            log.write(f"# {step['id']}: verified existing dependency {step['path']}\n")
            continue
        command = [resolve(str(value), manifest_sha256) for value in argv]
        log.write(f"# {step['id']}\n")
        log.write(shlex.join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=step["cwd"], check=True)
        if step["id"] == "build_sparse_input":
            if not manifest.is_file():
                raise FileNotFoundError(f"sparse input step did not create manifest: {manifest}")
            manifest_sha256 = digest(manifest)
PY
