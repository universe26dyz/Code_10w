#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
SUBJECT_ID=CYJ
BASELINE_EXP_ROOT="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
QUALITY_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1"
MLP_RUN="${BASELINE_EXP_ROOT}/runs/frozen_mlp_B6"
PREPROCESSED_ROOT="${BASELINE_EXP_ROOT}/inputs/preprocessed"
NATIVE_REFERENCE_ROOT="${BASELINE_EXP_ROOT}/inputs/reference/native_reference_2d"
OUTPUT_ROOT="${QUALITY_ROOT}/${SUBJECT_ID}/G0_geometry_correspondence_baseline_v2"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
COMMIT_SHA="$(git rev-parse HEAD)"

for required in \
  "${MLP_RUN}" "${PREPROCESSED_ROOT}" "${NATIVE_REFERENCE_ROOT}" \
  "${MLP_RUN}/t1_t2_native_plane_sax.npz" "${MLP_RUN}/t1_t2_native_plane_2ch.npz" "${MLP_RUN}/t1_t2_native_plane_4ch.npz" \
  "${NATIVE_REFERENCE_ROOT}/native_reference_manifest.json"; do
  [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 1; }
done
if [[ -e "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty G0 output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi

RUN_LOG="$(mktemp -t code10w-g0.XXXXXX.log)"
printf '%s\n' "G0 implementation revision: baseline_v2" "G0 Git commit: ${COMMIT_SHA}" | tee "${RUN_LOG}"
python quality_experiments/G0_geometry_correspondence/run_g0_geometry_audit.py \
  --subject-id "${SUBJECT_ID}" \
  --run-root "${MLP_RUN}" \
  --native-reference-root "${NATIVE_REFERENCE_ROOT}" \
  --preprocessed-root "${PREPROCESSED_ROOT}" \
  --output "${OUTPUT_ROOT}" 2>&1 | tee -a "${RUN_LOG}"
cp "${RUN_LOG}" "${OUTPUT_ROOT}/logs/run.log"
printf '%s\n' "G0 outputs: ${OUTPUT_ROOT}" "Summary: ${OUTPUT_ROOT}/RESULT_SUMMARY.md"
