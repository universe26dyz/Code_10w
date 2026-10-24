#!/usr/bin/env bash
# Formal SCMR one-stage Fit-first route. This wrapper is intentionally server-run only.
set -euo pipefail

: "${SUBJECT:?set SUBJECT}"
: "${PREPARED_ROOT:?set PREPARED_ROOT}"
: "${STANDARDIZED_NATIVE_DICTIONARY_ROOT:?set STANDARDIZED_NATIVE_DICTIONARY_ROOT}"
: "${STANDARDIZED_REUSE_MANIFEST:?set STANDARDIZED_REUSE_MANIFEST}"
: "${SPARSE_MANIFEST:?set SPARSE_MANIFEST}"
: "${EXPECTED_GENERATOR_GIT_SHA:?set EXPECTED_GENERATOR_GIT_SHA}"
: "${EXPECTED_MANIFEST_SHA256:?set EXPECTED_MANIFEST_SHA256}"
: "${NESVOR:?set NESVOR executable}"
: "${OUTPUT:?set fresh OUTPUT}"
: "${DEVICE:?set DEVICE, e.g. 0}"
: "${PYTHON:?set PYTHON}"

"${PYTHON}" -m paper_experiments.joint_vs_2dfit.phase2_clean.baseline \
  --prepared-root "${PREPARED_ROOT}" \
  --subject-id "${SUBJECT}" \
  --standardized-native-maps-root "${STANDARDIZED_NATIVE_DICTIONARY_ROOT}" \
  --standardized-reuse-manifest "${STANDARDIZED_REUSE_MANIFEST}" \
  --sparse-manifest "${SPARSE_MANIFEST}" \
  --expected-generator-git-sha "${EXPECTED_GENERATOR_GIT_SHA}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA256}" \
  --nesvor-executable "${NESVOR}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --execute
