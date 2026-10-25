"""Contracts for Final5 all/observed/held-out native reporting partitions."""
from __future__ import annotations

import json

import pytest


def _write_json(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle_roots(tmp_path, *, endpoint_manifest, endpoint_record) -> tuple:
    """Create only the manifests required by the native-bundle validator."""
    endpoint = tmp_path / "local_mirror" / "fit_first" / "CYJ" / "L100"
    _write_json(endpoint / "manifest.json", endpoint_manifest)
    direct = tmp_path / "bundles" / "proposed"
    fit = tmp_path / "bundles" / "fit_first"
    _write_json(direct / "manifest.json", {
        "schema": "direct_native_prediction_bundle/v1", "endpoint": "central",
    })
    _write_json(fit / "manifest.json", {
        "schema": "fit_first_native_prediction_bundle/v6",
        "endpoint": "central",
        "endpoint_source_manifest": endpoint_record,
    })
    return direct, fit, endpoint


def test_bundle_validation_accepts_relocated_endpoint_manifest_with_matching_sha(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_partitioned_native_metrics import _validate_bundles

    endpoint_manifest = {
        "schema": "phase4r_fixed_l100_fit_first_export_bundle/v2",
        "subject": "CYJ", "condition": "L100", "endpoint": "central",
    }
    local_endpoint = tmp_path / "local_mirror" / "fit_first" / "CYJ" / "L100" / "manifest.json"
    _write_json(local_endpoint, endpoint_manifest)
    declared = {
        "path": "/data/dengyz/dataset/Code_10w_runs/final5/05_formal_endpoints/central/fit_first/CYJ/L100/manifest.json",
        "sha256": sha256(local_endpoint),
    }
    direct, fit, endpoint = _bundle_roots(tmp_path, endpoint_manifest=endpoint_manifest, endpoint_record=declared)

    _direct, _fit, provenance = _validate_bundles(
        direct_root=direct, fit_root=fit, subject="CYJ", case="L100", fit_endpoint=endpoint)

    assert provenance["declared_original_path"] == declared["path"]
    assert provenance["actual_local_path"] == str(local_endpoint.resolve())
    assert provenance["sha256"] == declared["sha256"]
    assert provenance["relocated_for_local_mirror"] is True


@pytest.mark.parametrize(
    ("endpoint_manifest", "declared_sha256", "message"),
    [
        (None, "0" * 64, "SHA256 mismatch"),
        ({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "subject": "OTHER", "condition": "L100", "endpoint": "central"}, None,
         "schema/subject/case/endpoint mismatch"),
    ],
)
def test_bundle_validation_rejects_relocated_endpoint_with_wrong_hash_or_identity(
        tmp_path, endpoint_manifest, declared_sha256, message) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_partitioned_native_metrics import _validate_bundles

    manifest = endpoint_manifest or {
        "schema": "phase4r_fixed_l100_fit_first_export_bundle/v2",
        "subject": "CYJ", "condition": "L100", "endpoint": "central",
    }
    endpoint_file = tmp_path / "local_mirror" / "fit_first" / "CYJ" / "L100" / "manifest.json"
    _write_json(endpoint_file, manifest)
    declared = {
        "path": "/data/dengyz/dataset/Code_10w_runs/final5/05_formal_endpoints/central/fit_first/CYJ/L100/manifest.json",
        "sha256": declared_sha256 or sha256(endpoint_file),
    }
    direct, fit, endpoint = _bundle_roots(tmp_path, endpoint_manifest=manifest, endpoint_record=declared)

    with pytest.raises(ValueError, match=message):
        _validate_bundles(direct_root=direct, fit_root=fit, subject="CYJ", case="L100", fit_endpoint=endpoint)


def test_partitioned_roles_preserve_all_observed_and_heldout_sets() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_partitioned_native_metrics import partition_group_roles

    roles = {("sax", 0): "observed", ("2ch", 1): "observed", ("2ch", 2): "heldout"}
    assert partition_group_roles(roles) == {
        "all": {("sax", 0), ("2ch", 1), ("2ch", 2)},
        "observed": {("sax", 0), ("2ch", 1)},
        "heldout": {("2ch", 2)},
    }
