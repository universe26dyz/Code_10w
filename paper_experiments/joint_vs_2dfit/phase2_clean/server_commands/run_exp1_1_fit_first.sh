#!/usr/bin/env bash
# Server-only wrapper for the frozen-map CYJ Experiment 1.1 Fit-first route.
set -euo pipefail

: "${SUBJECT:?set SUBJECT}"
: "${STANDARDIZED_NATIVE_DICTIONARY_ROOT:?set STANDARDIZED_NATIVE_DICTIONARY_ROOT}"
: "${SPARSE_MANIFEST:?set SPARSE_MANIFEST}"
: "${EXPECTED_GENERATOR_GIT_SHA:?set EXPECTED_GENERATOR_GIT_SHA}"
: "${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256}"
: "${NESVOR:?set NESVOR executable}"
: "${OUTPUT:?set fresh OUTPUT}"
: "${DEVICE:?set DEVICE}"
: "${PYTHON:?set PYTHON}"

"${PYTHON}" -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_exp1_1 \
  --subject "${SUBJECT}" \
  --sparse-manifest "${SPARSE_MANIFEST}" \
  --standardized-native-dictionary-root "${STANDARDIZED_NATIVE_DICTIONARY_ROOT}" \
  --expected-generator-git-sha "${EXPECTED_GENERATOR_GIT_SHA}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --nesvor "${NESVOR}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --execute
