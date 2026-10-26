"""CPU-only contracts for direct-native endpoint-audit support semantics."""
from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
import torch

from paper_experiments.joint_vs_2dfit.phase2_clean import direct_native_endpoint_audit as audit


def test_direct_and_volume_metrics_share_one_exact_paired_support() -> None:
    reference = np.array([[1000.0, 1000.0, np.nan]])
    support = np.array([[True, True, True]])
    l100 = np.array([[True, True, True]])
    direct = np.array([[1001.0, np.nan, 999.0]])
    volume = np.array([[1002.0, 1003.0, 998.0]])
    shared = audit.exact_shared_support(reference, support, l100, direct, volume, np.ones((1, 3), bool), np.array([[True, False, True]]))
    assert shared.tolist() == [[True, False, False]]


def test_direct_path_is_model_inr_and_volume_path_only_samples_the_central_nifti() -> None:
    source = inspect.getsource(audit.run_direct_native_endpoint_audit)
    assert "direct_native_world_grid(model, space" in source
    assert "_model_prediction(model, world, field" in source
    assert "sample_exported_volume_at_train_world(central[parameter].get_fdata()" in source
    assert "native_affine" not in source
    assert "model.inr(points)" in inspect.getsource(audit._model_prediction)


def test_output_support_does_not_depend_on_a_private_method_mask() -> None:
    reference = np.ones((2, 2)); source = np.ones((2, 2), bool); l100 = np.ones((2, 2), bool)
    direct = np.ones((2, 2)); volume = np.ones((2, 2))
    direct_valid = np.array([[True, True], [False, True]])
    volume_valid = np.array([[True, False], [True, True]])
    support = audit.exact_shared_support(reference, source, l100, direct, volume, direct_valid, volume_valid)
    assert support.tolist() == [[True, False], [False, True]]


def test_volume_branch_converts_training_world_to_physical_ras_before_sampling() -> None:
    """A missing conversion samples outside this physical-RAS volume."""
    class Space:
        def train_ras_to_physical(self, values: torch.Tensor) -> torch.Tensor:
            return values + torch.tensor([10.0, 0.0, 0.0])

    volume = np.array([[[101.0]], [[202.0]]])
    train_world = np.zeros((1, 1, 3), dtype=np.float32)
    affine = np.eye(4); affine[0, 3] = 10.0
    values, valid = audit.sample_exported_volume_at_train_world(volume, affine, train_world, Space(), torch, order=1)
    assert valid.tolist() == [[True]]
    assert values.tolist() == [[101.0]]


def test_checkpoint_pose_restoration_prefers_post_stack_initialization_and_supports_legacy_field() -> None:
    """Catches rebuilding the hash-grid bbox from the pre-stack DICOM pose."""
    post = torch.tensor([[1., 2., 3., 4., 5., 6.]])
    legacy = torch.zeros((1, 6))

    assert torch.equal(audit._checkpoint_initial_pose({
        "group_axisangle_post_stack_init_physical": post,
        "group_axisangle_init_physical": legacy,
    }, torch, torch.device("cpu")), post)
    assert torch.equal(audit._checkpoint_initial_pose({"group_axisangle_init_physical": legacy}, torch, torch.device("cpu")), legacy)


def test_checkpoint_model_shape_validation_rejects_inr_encoding_mismatch_before_load() -> None:
    """Catches a strict load attempt after a differently sized hash encoding was built."""
    class Model:
        def state_dict(self):
            return {
                "inr.encoding.params": torch.zeros(4),
                "forward_model.quantitative_inr.encoding.params": torch.zeros(4),
            }

    checkpoint_state = {
        "inr.encoding.params": torch.zeros(5),
        "forward_model.quantitative_inr.encoding.params": torch.zeros(5),
    }
    with pytest.raises(ValueError, match="encoding parameter shape mismatch"):
        audit._validate_checkpoint_model_shapes(Model(), checkpoint_state)


def test_checkpoint_training_space_validation_rejects_bbox_mismatch() -> None:
    """Catches sampling coordinates from a current-input bbox rather than checkpoint space."""
    class Space:
        def state_dict(self):
            return {
                "center_ras_mm": torch.zeros(3), "spatial_scaling": 1.0,
                "physical_bbox_ras_mm": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
                "bbox_train": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
                "group_resolution_xyz_mm": torch.ones((1, 3)),
                "group_axisangle_init_physical": torch.zeros((1, 6)),
                "group_axisangle_init_train": torch.zeros((1, 6)),
            }

    saved = Space().state_dict(); saved["bbox_train"] = torch.tensor([[0., 0., 0.], [2., 1., 1.]])
    with pytest.raises(ValueError, match="bounding-box mismatch"):
        audit._validate_rebuilt_training_space(Space(), saved, torch)


def test_checkpoint_training_space_validation_compares_cpu_copies() -> None:
    """Checkpoint tensors may be restored on CUDA while rebuilt state is CPU."""
    class Tensor:
        shape = (1,)

        def __init__(self, device: str): self.device = device
        def detach(self): return self
        def cpu(self): return Tensor("cpu")

    class Torch:
        @staticmethod
        def as_tensor(value): return value

        @staticmethod
        def allclose(expected, actual, **_kwargs):
            assert expected.device == actual.device == "cpu"
            return True

    values = {key: Tensor("cuda") for key in (
        "center_ras_mm", "spatial_scaling", "physical_bbox_ras_mm", "bbox_train",
        "group_resolution_xyz_mm", "group_axisangle_init_physical", "group_axisangle_init_train",
    )}

    class Space:
        def state_dict(self): return {key: Tensor("cpu") for key in values}

    audit._validate_rebuilt_training_space(Space(), values, Torch)


def test_prepared_input_identity_uses_training_manifest_hash_not_local_path(tmp_path) -> None:
    """Catches using a same-named but content-different local observation file."""
    observations = []
    for stack, content in (("sax", b"sax"), ("2ch", b"2ch"), ("4ch", b"4ch")):
        path = tmp_path / "local_mirror" / stack / "observations.npz"
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content); observations.append(path)
    checkpoint = tmp_path / "run" / "model.pt"; checkpoint.parent.mkdir(); checkpoint.touch()
    manifest = {
        "prepared_inputs": [f"/data/server/DYZ/{stack}/observations.npz" for stack in ("sax", "2ch", "4ch")],
        "prepared_manifest_sha256": audit._prepared_inputs_sha256(observations),
    }
    (checkpoint.parent / "experiment_manifest.json").write_text(json.dumps(manifest))

    audit._validate_prepared_input_identity(checkpoint, observations)
    observations[0].write_bytes(b"different")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit._validate_prepared_input_identity(checkpoint, observations)


def test_paths_accepts_dyz_subject_root_without_cyj_specific_error(tmp_path) -> None:
    """Catches the legacy CYJ-only fallback and diagnostic for other Final5 subjects."""
    for stack in ("sax", "2ch", "4ch"):
        path = tmp_path / "DYZ" / stack / "observations.npz"
        path.parent.mkdir(parents=True, exist_ok=True); path.touch()

    assert audit._paths(tmp_path) == [tmp_path / "DYZ" / stack / "observations.npz" for stack in ("sax", "2ch", "4ch")]


def test_load_model_rebuilds_with_checkpoint_post_stack_pose_and_returns_saved_space(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catches a load path that validates shapes but still derives coordinates from a rebuilt space."""
    prepared = []
    for stack in ("sax", "2ch", "4ch"):
        path = tmp_path / stack / "observations.npz"; path.parent.mkdir(exist_ok=True); path.write_bytes(stack.encode()); prepared.append(path)
    checkpoint = tmp_path / "model.pt"; checkpoint.touch()
    (tmp_path / "experiment_manifest.json").write_text(json.dumps({
        "prepared_inputs": [f"/server/DYZ/{stack}/observations.npz" for stack in ("sax", "2ch", "4ch")],
        "prepared_manifest_sha256": audit._prepared_inputs_sha256(prepared),
    }))
    training_space = {
        "center_ras_mm": torch.zeros(3), "spatial_scaling": 1.0,
        "physical_bbox_ras_mm": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
        "bbox_train": torch.tensor([[0., 0., 0.], [1., 1., 1.]]),
        "group_resolution_xyz_mm": torch.ones((1, 3)),
        "group_axisangle_init_physical": torch.ones((1, 6)),
        "group_axisangle_init_train": torch.ones((1, 6)),
    }
    class Model:
        def state_dict(self):
            return {"inr.encoding.params": torch.zeros(4), "forward_model.quantitative_inr.encoding.params": torch.zeros(4)}
        def eval(self): return self
    class Dataset:
        def __init__(self, *_args, **_kwargs): pass
    class RebuiltSpace:
        def state_dict(self): return training_space
    payload = {"resolved_config": {"decoder": {}}, "training_space": training_space,
               "group_axisangle_post_stack_init_physical": torch.ones((1, 6)),
               "group_axisangle_init_physical": torch.zeros((1, 6)),
               "model_state": Model().state_dict()}
    captured: list[torch.Tensor] = []
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr("trad.modules.module_03_dataset_geometry.quantitative_point_dataset.QuantPointDataset", Dataset)
    monkeypatch.setattr("mlp.modules.module_07_objective_training.mlp_trainer.build_mlp_training_model",
                        lambda _dataset, _config, _protocol, _device, *, initial_group_axisangle_physical: captured.append(initial_group_axisangle_physical) or (Model(), RebuiltSpace(), None, None))
    monkeypatch.setattr("reconstruction_core.orchestration.load_checkpoint", lambda *_args, **_kwargs: payload)

    _model, restored_space, _torch = audit._load_model(checkpoint, prepared, "cpu")

    assert torch.equal(captured[0], payload["group_axisangle_post_stack_init_physical"])
    assert torch.equal(restored_space.bbox_train, training_space["bbox_train"])


def _sparse_manifest(path, *, subject: str = "DYZ", case: str = "L100", generator: str = "a" * 40,
                     active: tuple[str, ...] = ("sax", "2ch", "4ch")) -> None:
    path.write_text(json.dumps({
        "schema_version": "phase4r_sparse_profile_v2", "status": "ELIGIBLE",
        "subject": subject, "case_id": case, "git_sha": generator,
        "stacks": {stack: {"training_stack_present": stack in active} for stack in ("sax", "2ch", "4ch")},
    }))


def test_q002_manifest_validates_mirrored_inputs_by_stack_hash_and_sparse_provenance(tmp_path) -> None:
    """Catches rejecting the actual Q002 manifest merely because its paths are server paths."""
    inputs = []
    for stack in ("sax", "2ch", "4ch"):
        path = tmp_path / "local" / stack / "observations.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stack.encode()); inputs.append(path)
    sparse = tmp_path / "sparse.json"; _sparse_manifest(sparse)
    checkpoint = tmp_path / "run" / "model.pt"; checkpoint.parent.mkdir(); checkpoint.touch()
    records = {f"/data/server/DYZ/{path.parent.name}/observations.npz": audit.sha256(path) for path in inputs}
    (checkpoint.parent / "experiment_manifest.json").write_text(json.dumps({
        "subject_id": "DYZ", "case_id": "L100", "prepared_inputs_sha256": records,
        "training_input_sha256": dict(records),
        "sparse_manifest": {"path": "/data/server/DYZ/sparse_manifest.json", "sha256": audit.sha256(sparse), "generator_git_sha": "a" * 40},
    }))

    active = audit._sparse_contract(sparse)
    audit._validate_prepared_input_identity(checkpoint, inputs, active["active_stacks"])
    audit._validate_checkpoint_sparse_provenance(checkpoint, active)


def test_r3_l100_manifest_uses_per_stack_hashes_without_profile_sparse_record(tmp_path) -> None:
    """R3 L100 is full-input identity data, not a profile-v2 sparse case."""
    inputs = []
    for stack in ("sax", "2ch", "4ch"):
        path = tmp_path / "local" / stack / "observations.npz"; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stack.encode()); inputs.append(path)
    sparse = tmp_path / "sparse.json"
    sparse.write_text(json.dumps({
        "schema_version": "phase4r_sparse_v1", "status": "ELIGIBLE", "subject": "DYZ",
        "condition": "L100", "phase": 0, "git_sha": "d" * 40,
        "stacks": {stack: {} for stack in ("sax", "2ch", "4ch")},
    }))
    checkpoint = tmp_path / "run" / "model.pt"; checkpoint.parent.mkdir(); checkpoint.touch()
    records = {f"/data/server/DYZ/{path.parent.name}/observations.npz": audit.sha256(path) for path in inputs}
    (checkpoint.parent / "experiment_manifest.json").write_text(json.dumps({
        "subject_id": "DYZ", "prepared_inputs_sha256": records,
    }))

    contract = audit._sparse_contract(sparse)

    assert contract["active_stacks"] == ("sax", "2ch", "4ch")
    audit._validate_prepared_input_identity(checkpoint, inputs, contract["active_stacks"])
    audit._validate_checkpoint_sparse_provenance(checkpoint, contract)


def test_sparse_case_accepts_only_manifest_active_stack_subset_in_canonical_order(tmp_path) -> None:
    """Catches treating absent 2CH/4CH sparse stacks as mandatory audit inputs."""
    sparse = tmp_path / "sparse.json"; _sparse_manifest(sparse, case="SAX100_2CH0_4CH0", active=("sax",))
    path = tmp_path / "DYZ" / "sax" / "observations.npz"; path.parent.mkdir(parents=True); path.write_bytes(b"sax")

    contract = audit._sparse_contract(sparse)

    assert contract["active_stacks"] == ("sax",)
    assert audit._paths(tmp_path, active_stacks=contract["active_stacks"]) == [path]


def test_q002_input_or_sparse_provenance_mismatch_is_rejected(tmp_path) -> None:
    """Catches accepting a mismatched active input or a different sparse profile."""
    path = tmp_path / "sax" / "observations.npz"; path.parent.mkdir(parents=True); path.write_bytes(b"sax")
    sparse = tmp_path / "sparse.json"; _sparse_manifest(sparse, case="SAX100_2CH0_4CH0", active=("sax",))
    checkpoint = tmp_path / "run" / "model.pt"; checkpoint.parent.mkdir(); checkpoint.touch()
    (checkpoint.parent / "experiment_manifest.json").write_text(json.dumps({
        "subject_id": "DYZ", "case_id": "SAX100_2CH0_4CH0",
        "prepared_inputs_sha256": {"/server/DYZ/sax/observations.npz": "0" * 64},
        "sparse_manifest": {"path": "/server/sparse.json", "sha256": audit.sha256(sparse), "generator_git_sha": "b" * 40},
    }))
    contract = audit._sparse_contract(sparse)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        audit._validate_prepared_input_identity(checkpoint, [path], contract["active_stacks"])
    with pytest.raises(ValueError, match="generator"):
        audit._validate_checkpoint_sparse_provenance(checkpoint, contract)


@pytest.mark.parametrize(
    ("manifest_change", "message"),
    [
        ({"subject_id": "CYJ"}, "subject/case"),
        ({"case_id": "L100"}, "subject/case"),
        ({"sparse_sha": "0" * 64}, "SHA256"),
    ],
)
def test_q002_sparse_provenance_requires_subject_case_and_manifest_hash(tmp_path, manifest_change, message) -> None:
    """Catches binding a checkpoint to a different sparse subject, case, or manifest."""
    sparse = tmp_path / "sparse.json"; _sparse_manifest(sparse, case="SAX100_2CH0_4CH0", active=("sax",))
    checkpoint = tmp_path / "run" / "model.pt"; checkpoint.parent.mkdir(); checkpoint.touch()
    record = {"path": "/server/sparse.json", "sha256": audit.sha256(sparse), "generator_git_sha": "a" * 40}
    if "sparse_sha" in manifest_change:
        record["sha256"] = manifest_change["sparse_sha"]
    (checkpoint.parent / "experiment_manifest.json").write_text(json.dumps({
        "subject_id": manifest_change.get("subject_id", "DYZ"),
        "case_id": manifest_change.get("case_id", "SAX100_2CH0_4CH0"),
        "sparse_manifest": record,
    }))

    with pytest.raises(ValueError, match=message):
        audit._validate_checkpoint_sparse_provenance(checkpoint, audit._sparse_contract(sparse))
