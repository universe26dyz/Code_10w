#!/usr/bin/env bash
set -euo pipefail
LOCAL_REFERENCE_ROOT=/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_native_reference_v2
SERVER_HOST="${Q001_SERVER_HOST:?Set the formal server host.}"
SERVER_REFERENCE_ROOT=/data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/Q001_full_fov_native_reference_v2
[[ -f "${LOCAL_REFERENCE_ROOT}/native_reference_manifest.json" ]] || { echo "Missing local Q001 native-reference manifest" >&2; exit 1; }
# This is intentionally a pre-rsync check: no merge, overwrite, or deletion is allowed.
ssh "${SERVER_HOST}" "test ! -e '${SERVER_REFERENCE_ROOT}' || test -z \"\$(find '${SERVER_REFERENCE_ROOT}' -mindepth 1 -maxdepth 1 -print -quit)\"" || { echo "Refusing non-empty remote Q001 reference destination: ${SERVER_REFERENCE_ROOT}" >&2; exit 1; }
rsync -a --checksum --protect-args "${LOCAL_REFERENCE_ROOT}/" "${SERVER_HOST}:${SERVER_REFERENCE_ROOT}/"
printf 'Server verification: Q001_REFERENCE_ROOT=%q Q001_INPUT_ROOT=%q bash quality_experiments/server_commands/verify_Q001_reference_CYJ.sh\n' "${SERVER_REFERENCE_ROOT}" /data/dengyz/dataset/Code_10w_v1/Q001_full_fov_inputs_v2
