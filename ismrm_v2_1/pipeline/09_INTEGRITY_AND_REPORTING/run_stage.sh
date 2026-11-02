#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${1:-python}
"$PYTHON" "$PROJECT_ROOT/scripts/verify_project_integrity.py" "$PROJECT_ROOT"
"$PYTHON" "$PROJECT_ROOT/scripts/verify_no_legacy_code_path.py" "$PROJECT_ROOT"
"$PYTHON" "$PROJECT_ROOT/scripts/verify_runtime_contract.py" --root "$PROJECT_ROOT"

