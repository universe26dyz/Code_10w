import csv
import pytest
import torch

from mlp.modules.module_05_signal_decoder import train_mlp as train_module
from mlp.modules.module_05_signal_decoder.synthetic_dataset import generate_rr_synthetic_dataset


def _dataset(tmp_path):
    dataset = tmp_path / "rr"
    generate_rr_synthetic_dataset(dataset, {
        "n_rhythms": 6, "tissue_samples_per_rhythm": 4,
        "split_rhythms": {"train": 2, "valid": 2, "test": 2}, "seed": 13,
        "store_jacobian": False, "teacher_chunk_size": 4, "teacher_device": "cpu",
    })
    return dataset


def _config(epochs, **training_overrides):
    training = {"device": "cpu", "seed": 17, "epochs": epochs, "batch_size": 4,
                "learning_rate": 1e-3, "scheduler_step_size": 2, "scheduler_gamma": 0.5,
                "gradient_samples": 2}
    training.update(training_overrides)
    return {
        "training": training,
        "mlp_loss": {"signal_mse_weight": 1.0, "jacobian_weight": 0.0, "cosine_weight": 0.0},
    }


def _history(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _train(tmp_path, epochs, *, output_name="run", resume=None, **training_overrides):
    dataset = _dataset(tmp_path) if not (tmp_path / "rr").exists() else tmp_path / "rr"
    output = tmp_path / output_name
    return train_module.train_mlp(dataset, _config(epochs, **training_overrides), "mlp/configs/protocol_hhz_v1.yaml", output, resume=resume), dataset, output


def test_training_uses_ram_splits_and_writes_incremental_history_and_last_checkpoint(monkeypatch, tmp_path):
    seen = []
    original = train_module.load_h5_split
    original_save = torch.save

    def load_ram(path, split, **kwargs):
        seen.append(split)
        return original(path, split, **kwargs)

    def reject_lazy(*_args, **_kwargs):
        raise AssertionError("active training must not instantiate LazyRRSplit")

    def save_checkpoint(value, path, *args, **kwargs):
        if str(path).endswith("signal_simulator_last.pth") and value["epoch"] == 1:
            assert [int(row["epoch"]) for row in _history(tmp_path / "run" / "train_history.csv")] == [1]
        return original_save(value, path, *args, **kwargs)

    monkeypatch.setattr(train_module, "load_h5_split", load_ram)
    monkeypatch.setattr(train_module, "LazyRRSplit", reject_lazy)
    monkeypatch.setattr(train_module.torch, "save", save_checkpoint)
    _, dataset, output = _train(tmp_path, 3)

    assert seen == ["train", "valid", "test"]
    rows = _history(output / "train_history.csv")
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert set(rows[0]) == {"epoch", "train_mse", "valid_mse", "lr", "epoch_seconds", "best_valid_mse", "best_epoch", "is_best"}
    last = torch.load(output / "signal_simulator_last.pth", map_location="cpu", weights_only=False)
    assert last["epoch"] == 3
    assert {"optimizer_state_dict", "scheduler_state_dict", "best_epoch", "best_valid_mse", "dataset_metadata_sha256", "python_random_state", "numpy_random_state", "torch_rng_state"}.issubset(last)
    assert last["training_contract"] == {"batch_size": 4, "learning_rate": 1e-3, "optimizer": "Adam", "scheduler": "StepLR", "scheduler_step_size": 2, "scheduler_gamma": 0.5}
    assert last["dataset_metadata_sha256"] == train_module.sha256_file(dataset / "dataset_metadata.json")
    resolved = (output / "mlp_config_resolved.yaml").read_text(encoding="utf-8")
    assert "dataset_metadata_sha256" in resolved and "completed_epochs: 3" in resolved


def test_resume_appends_history_and_restores_training_state_and_rng(tmp_path):
    _, _, resumed_output = _train(tmp_path, 2, output_name="resumed")
    resume = resumed_output / "signal_simulator_last.pth"
    _train(tmp_path, 4, output_name="resumed", resume=resume)
    _, _, fresh_output = _train(tmp_path, 4, output_name="fresh")

    rows = _history(resumed_output / "train_history.csv")
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3, 4]
    resumed_last = torch.load(resumed_output / "signal_simulator_last.pth", map_location="cpu", weights_only=False)
    fresh_last = torch.load(fresh_output / "signal_simulator_last.pth", map_location="cpu", weights_only=False)
    assert resumed_last["epoch"] == 4
    assert resumed_last["scheduler_state_dict"] == fresh_last["scheduler_state_dict"]
    for key, value in fresh_last["state_dict"].items():
        torch.testing.assert_close(resumed_last["state_dict"][key], value, rtol=0, atol=0)
    assert "resumed_from_epoch: 2" in (resumed_output / "mlp_config_resolved.yaml").read_text(encoding="utf-8")


def test_resume_discards_one_uncheckpointed_history_epoch_and_retrains_it(tmp_path):
    _, _, output = _train(tmp_path, 2)
    resume, history = output / "signal_simulator_last.pth", output / "train_history.csv"
    rows = _history(history)
    interrupted = dict(rows[-1]); interrupted.update({"epoch": "3", "train_mse": "uncheckpointed"})
    with history.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=list(interrupted)).writerow(interrupted)

    _train(tmp_path, 4, resume=resume)

    rows = _history(history)
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3, 4]
    assert rows[2]["train_mse"] != "uncheckpointed"


@pytest.mark.parametrize(("field", "value", "message"), [
    ("dataset_metadata_sha256", "0" * 64, "dataset_metadata_sha256"),
    ("dataset_schema", "legacy", "dataset_schema"),
    ("dataset_split_mode", "subject", "dataset_split_mode"),
    ("architecture", "12-16-10", "architecture"),
    ("protocol_hhz_v1", {"tr_ms": 3.2, "vps": 87}, "protocol_hhz_v1"),
    ("parameter_ranges", {"t1_ms": [1, 2]}, "parameter_ranges"),
    ("seed", 99, "seed"),
    ("mlp_loss", {"signal_mse_weight": 1.0, "jacobian_weight": 1.0, "cosine_weight": 0.0}, "mlp_loss"),
])
def test_resume_rejects_incompatible_checkpoint(tmp_path, field, value, message):
    _, _, output = _train(tmp_path, 2)
    resume = output / "signal_simulator_last.pth"
    checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
    checkpoint[field] = value
    torch.save(checkpoint, resume)
    with pytest.raises(ValueError, match=message):
        _train(tmp_path, 4, resume=resume)


def test_resume_rejects_history_mismatch_and_completed_checkpoint(tmp_path):
    _, _, output = _train(tmp_path, 2)
    resume = output / "signal_simulator_last.pth"
    history = output / "train_history.csv"
    history.write_text("\n".join(history.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="history"):
        _train(tmp_path, 4, resume=resume)

    _, _, too_long_output = _train(tmp_path, 2, output_name="too_long")
    too_long_history = too_long_output / "train_history.csv"
    rows = _history(too_long_history)
    with too_long_history.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[-1]))
        for epoch in (3, 4):
            row = dict(rows[-1]); row["epoch"] = epoch; writer.writerow(row)
    with pytest.raises(ValueError, match="history"):
        _train(tmp_path, 5, output_name="too_long", resume=too_long_output / "signal_simulator_last.pth")

    _, _, complete_output = _train(tmp_path, 2, output_name="complete")
    with pytest.raises(ValueError, match="configured total epochs"):
        _train(tmp_path, 2, output_name="complete", resume=complete_output / "signal_simulator_last.pth")


@pytest.mark.parametrize("epochs", [[1, 3], [1, 2, 2]])
def test_resume_rejects_non_contiguous_or_duplicate_history(tmp_path, epochs):
    _, _, output = _train(tmp_path, 2)
    history = output / "train_history.csv"
    source_rows = _history(history)
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0])); writer.writeheader()
        for epoch in epochs:
            row = dict(source_rows[min(epoch, 2) - 1]); row["epoch"] = epoch; writer.writerow(row)
    with pytest.raises(ValueError, match="history"):
        _train(tmp_path, 4, resume=output / "signal_simulator_last.pth")


@pytest.mark.parametrize("training_overrides", [
    {"batch_size": 2}, {"learning_rate": 2e-3}, {"scheduler_step_size": 1}, {"scheduler_gamma": 0.25},
])
def test_resume_rejects_changed_optimizer_training_contract(tmp_path, training_overrides):
    _, _, output = _train(tmp_path, 2)
    with pytest.raises(ValueError, match="training_contract"):
        _train(tmp_path, 4, resume=output / "signal_simulator_last.pth", **training_overrides)
