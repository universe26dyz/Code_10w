#!/usr/bin/env bash
set -euo pipefail
mode=${1:---dry-run}; shift || true
[[ "$mode" == "--dry-run" || "$mode" == "--execute" ]] || { echo "usage: $0 [--dry-run|--execute] --selected-case-id ID --profile-id ID --subject-id ID --prepared-root ROOT --sparse-manifest PATH --expected-manifest-sha256 SHA --expected-generator-git-sha SHA --b6-model PATH --expected-b6-sha256 SHA --decoder PATH --protocol PATH --expected-protocol-sha256 SHA --output ROOT" >&2; exit 2; }
selected_case_id= profile_id= subject_id= prepared_root= sparse_manifest= expected_manifest_sha= expected_generator_sha= b6_model= expected_b6_sha= decoder= protocol= expected_protocol_sha= output=
while [[ $# -gt 0 ]]; do
    case "$1" in
        --selected-case-id) selected_case_id=${2:?}; shift 2;; --profile-id) profile_id=${2:?}; shift 2;; --subject-id) subject_id=${2:?}; shift 2;; --prepared-root) prepared_root=${2:?}; shift 2;; --sparse-manifest) sparse_manifest=${2:?}; shift 2;; --expected-manifest-sha256) expected_manifest_sha=${2:?}; shift 2;; --expected-generator-git-sha) expected_generator_sha=${2:?}; shift 2;; --b6-model) b6_model=${2:?}; shift 2;; --expected-b6-sha256) expected_b6_sha=${2:?}; shift 2;; --decoder) decoder=${2:?}; shift 2;; --protocol) protocol=${2:?}; shift 2;; --expected-protocol-sha256) expected_protocol_sha=${2:?}; shift 2;; --output) output=${2:?}; shift 2;; *) echo "unknown argument: $1" >&2; exit 2;;
    esac
done
for required in selected_case_id profile_id subject_id prepared_root sparse_manifest expected_manifest_sha expected_generator_sha b6_model expected_b6_sha decoder protocol expected_protocol_sha output; do [[ -n ${!required} ]] || { echo "missing --${required//_/-}" >&2; exit 2; }; done
[[ ! -e "$output" ]] || { echo "formal Q005 output must be fresh: $output" >&2; exit 2; }
python=${PYTHON:-python}
"$python" - "$selected_case_id" "$profile_id" "$subject_id" "$prepared_root" "$sparse_manifest" "$expected_manifest_sha" "$expected_generator_sha" "$b6_model" "$expected_b6_sha" "$decoder" "$protocol" "$expected_protocol_sha" "$output" <<'PY'
import sys
from argparse import Namespace
from pathlib import Path
from paper_experiments.joint_vs_2dfit.phase2_clean.sparse_profile_manifest import validate_sparse_profile_manifest
from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q005S640_K4_EXPERIMENT_ID
from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction import validate_formal_sparse_reconstruction_dependencies
(case_id, profile_id, subject, prepared, manifest, manifest_sha, generator_sha, b6, b6_sha, decoder, protocol, protocol_sha, output) = sys.argv[1:]
profile = validate_sparse_profile_manifest(manifest, subject=subject, strict=True, expected_generator_git_sha=generator_sha, expected_manifest_sha256=manifest_sha)
if profile["case_id"] != case_id or profile["profile_id"] != profile_id or Path(profile["prepared_root"]).resolve() != Path(prepared).resolve(): raise ValueError("selected case/profile/prepared root disagrees with sparse manifest")
validate_formal_sparse_reconstruction_dependencies(Namespace(experiment_id=Q005S640_K4_EXPERIMENT_ID, sparse_manifest=manifest, expected_sparse_generator_git_sha=generator_sha, expected_sparse_manifest_sha256=manifest_sha, b6_model=b6, expected_b6_model_sha256=b6_sha, signal_simulator=decoder, protocol=protocol, expected_protocol_sha256=protocol_sha, subject_id=subject, cropped_prepared_root=prepared, output=output))
PY
command=("$python" -m quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction --experiment-id Q005S640_K4_B6_roi_same_anchor_mse --cropped-prepared-root "$prepared_root" --subject-id "$subject_id" --sparse-manifest "$sparse_manifest" --expected-sparse-manifest-sha256 "$expected_manifest_sha" --expected-sparse-generator-git-sha "$expected_generator_sha" --b6-model "$b6_model" --expected-b6-model-sha256 "$expected_b6_sha" --signal-simulator "$decoder" --protocol "$protocol" --expected-protocol-sha256 "$expected_protocol_sha" --anchor-batch-size 640 --output "$output" --device cuda:0)
printf 'validated formal Q005 K4 admission; case=%s profile=%s\nfuture command: ' "$selected_case_id" "$profile_id"; printf '%q ' "${command[@]}"; printf '\n'
[[ "$mode" == "--dry-run" ]] && exit 0
"${command[@]}"
for required in model.pt experiment_manifest.json q002_route_manifest.json RESULT_SUMMARY.md; do [[ -f "$output/$required" ]] || { echo "formal Q005 postcheck missing $required" >&2; exit 1; }; done
