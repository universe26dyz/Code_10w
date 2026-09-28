#!/usr/bin/env bash
set -euo pipefail

CODE10W_ROOT=/data/dengyz/code/Code_10w
DATA_ROOT=/data/dengyz/dataset/Code_10w_v1
SUBJECT_ID=CYJ
BASELINE_EXP_ROOT="${DATA_ROOT}/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1"
QUALITY_ROOT="${DATA_ROOT}/Code_10w_runs/quality_experiments_v1"
TRAD_RUN="${BASELINE_EXP_ROOT}/runs/trad_B6"
MLP_RUN="${BASELINE_EXP_ROOT}/runs/frozen_mlp_B6"
TRAD_EVAL="${BASELINE_EXP_ROOT}/evaluation/trad_B6_map_domain_psf"
MLP_EVAL="${BASELINE_EXP_ROOT}/evaluation/frozen_mlp_B6_map_domain_psf"
PREPARED_ROOT="${BASELINE_EXP_ROOT}/inputs/prepared"
PREPROCESSED_ROOT="${BASELINE_EXP_ROOT}/inputs/preprocessed"
NATIVE_REFERENCE_ROOT="${BASELINE_EXP_ROOT}/inputs/reference/native_reference_2d"
MYOCARDIUM_MASK_ROOT="${BASELINE_EXP_ROOT}/inputs/reference/myocardium_masks_v2"
OUTPUT_ROOT="${QUALITY_ROOT}/${SUBJECT_ID}/D1_psf_smoothness_decomposition_baseline_v2"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cr_dreme
cd "${CODE10W_ROOT}"
COMMIT_SHA="$(git rev-parse HEAD)"

for required in \
  "${TRAD_RUN}" "${MLP_RUN}" "${TRAD_EVAL}" "${MLP_EVAL}" \
  "${PREPARED_ROOT}" "${PREPROCESSED_ROOT}" "${NATIVE_REFERENCE_ROOT}" "${MYOCARDIUM_MASK_ROOT}" \
  "${TRAD_RUN}/t1_t2_native_plane_sax.npz" "${TRAD_RUN}/t1_t2_native_plane_2ch.npz" "${TRAD_RUN}/t1_t2_native_plane_4ch.npz" \
  "${MLP_RUN}/t1_t2_native_plane_sax.npz" "${MLP_RUN}/t1_t2_native_plane_2ch.npz" "${MLP_RUN}/t1_t2_native_plane_4ch.npz" \
  "${TRAD_EVAL}/T1_native_map_domain_psf_comparison.npz" "${TRAD_EVAL}/T2_native_map_domain_psf_comparison.npz" \
  "${MLP_EVAL}/T1_native_map_domain_psf_comparison.npz" "${MLP_EVAL}/T2_native_map_domain_psf_comparison.npz"; do
  [[ -e "${required}" ]] || { echo "Missing required input: ${required}" >&2; exit 1; }
done

if [[ -e "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty D1 output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi

RUN_LOG="$(mktemp -t code10w-d1.XXXXXX.log)"
printf '%s\n' "D1 implementation revision: baseline_v2" "Git commit: ${COMMIT_SHA}" | tee "${RUN_LOG}"
python quality_experiments/D1_psf_smoothness_decomposition/run_d1_psf_smoothness_decomposition.py \
  --subject-id "${SUBJECT_ID}" \
  --trad-run "${TRAD_RUN}" \
  --mlp-run "${MLP_RUN}" \
  --trad-eval "${TRAD_EVAL}" \
  --mlp-eval "${MLP_EVAL}" \
  --preprocessed-root "${PREPROCESSED_ROOT}" \
  --native-reference-root "${NATIVE_REFERENCE_ROOT}" \
  --myocardium-mask-root "${MYOCARDIUM_MASK_ROOT}" \
  --output "${OUTPUT_ROOT}" 2>&1 | tee -a "${RUN_LOG}"
cp "${RUN_LOG}" "${OUTPUT_ROOT}/logs/run.log"
printf '%s\n' "D1 outputs: ${OUTPUT_ROOT}" "Metrics: ${OUTPUT_ROOT}/metrics" "Summary: ${OUTPUT_ROOT}/RESULT_SUMMARY.md"
