#!/usr/bin/env bash
# Subject-general immutable reporting wrapper; no reconstruction is launched.
set -euo pipefail

[[ $# -ge 2 && -n ${1:-} ]] || { echo "usage: $0 SUBJECT aggregate|figures ..." >&2; exit 2; }
SUBJECT=$1
shift
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/finalize_cyj_reporting.sh" --subject "$SUBJECT" "$@"
