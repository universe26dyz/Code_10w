import pytest
import torch
import json
from pathlib import Path
from types import SimpleNamespace

from modules.module_03_dataset_geometry import quantitative_point_dataset as dataset_module
from modules.module_03_dataset_geometry.quantitative_point_dataset import robust_trimmed_mean_intensity
from reconstruction_core.orchestration import _intensity_normalization_provenance
from modules.module_07_objective_training.experiment_infrastructure import write_experiment_manifest


def test_trimmed_mean_intensity_matches_strict_q10_q90_mask():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    scale = robust_trimmed_mean_intensity(values, lower_quantile=0.1, upper_quantile=0.9)
    expected = values[(values > torch.quantile(values, 0.1)) & (values < torch.quantile(values, 0.9))].mean()
    torch.testing.assert_close(scale, expected)
    assert torch.isfinite(scale) and scale > 0


def test_trimmed_mean_intensity_is_scale_invariant_and_rejects_empty_trim():
    values, factor = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0]), 17.0
    scale_one = robust_trimmed_mean_intensity(values, lower_quantile=0.1, upper_quantile=0.9)
    scale_two = robust_trimmed_mean_intensity(values * factor, lower_quantile=0.1, upper_quantile=0.9)
    torch.testing.assert_close(scale_two, scale_one * factor)
    torch.testing.assert_close(values / scale_one, values * factor / scale_two)
    with pytest.raises(ValueError, match="trimmed intensity set is empty"):
        robust_trimmed_mean_intensity(torch.ones(8), lower_quantile=0.1, upper_quantile=0.9)


def test_trimmed_mean_small_tensor_retains_exact_native_quantile_semantics():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0], dtype=torch.float64)
    scale, provenance = robust_trimmed_mean_intensity(
        values, lower_quantile=0.1, upper_quantile=0.9, return_provenance=True
    )
    q_low, q_high = torch.quantile(values, 0.1), torch.quantile(values, 0.9)
    expected = values[(values > q_low) & (values < q_high)].mean()
    torch.testing.assert_close(scale, expected)
    assert provenance == {
        "quantile_execution": "native_device",
        "input_numel": values.numel(),
        "lower_quantile": 0.1,
        "upper_quantile": 0.9,
    }


def test_trimmed_mean_exact_cpu_fallback_preserves_device_dtype_and_strict_thresholds(monkeypatch):
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0], dtype=torch.float64)
    original_quantile = torch.quantile
    calls = []

    def quantile_with_large_native_failure(input_values, quantile, *args, **kwargs):
        calls.append(input_values.device.type)
        if len(calls) == 1:
            raise RuntimeError("quantile() input tensor is too large")
        return original_quantile(input_values, quantile, *args, **kwargs)

    monkeypatch.setattr(dataset_module.torch, "quantile", quantile_with_large_native_failure)
    scale, provenance = robust_trimmed_mean_intensity(
        values, lower_quantile=0.1, upper_quantile=0.9, return_provenance=True
    )
    q_low, q_high = original_quantile(values, 0.1), original_quantile(values, 0.9)
    expected = values[(values > q_low) & (values < q_high)].mean()
    torch.testing.assert_close(scale, expected)
    assert scale.device == values.device and scale.dtype == values.dtype
    assert provenance["quantile_execution"] == "cpu_exact_fallback_for_large_tensor"
    assert provenance["input_numel"] == values.numel()
    assert calls == ["cpu", "cpu", "cpu"]


def test_trimmed_mean_does_not_swallow_unrelated_quantile_runtime_errors(monkeypatch):
    def quantile_other_failure(*args, **kwargs):
        raise RuntimeError("some other error")

    monkeypatch.setattr(dataset_module.torch, "quantile", quantile_other_failure)
    with pytest.raises(RuntimeError, match="some other error"):
        robust_trimmed_mean_intensity(torch.tensor([1.0, 2.0, 3.0]), lower_quantile=0.1, upper_quantile=0.9)


def test_normalization_provenance_records_execution_path_and_input_numel(monkeypatch):
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    training = {"intensity_normalization": {"enabled": True, "method": "trimmed_mean", "lower_quantile": 0.1, "upper_quantile": 0.9}}
    native = _intensity_normalization_provenance(training, SimpleNamespace(v=values))
    assert native["quantile_execution"] == "native_device"
    assert native["input_numel"] == values.numel()
    original_quantile, calls = torch.quantile, []

    def fail_once(input_values, quantile, *args, **kwargs):
        calls.append(quantile)
        if len(calls) == 1:
            raise RuntimeError("quantile() input tensor is too large")
        return original_quantile(input_values, quantile, *args, **kwargs)

    monkeypatch.setattr(dataset_module.torch, "quantile", fail_once)
    fallback = _intensity_normalization_provenance(training, SimpleNamespace(v=values))
    assert fallback["quantile_execution"] == "cpu_exact_fallback_for_large_tensor"
    assert fallback["input_numel"] == values.numel()
    torch.testing.assert_close(torch.tensor(fallback["scale"]), torch.tensor(native["scale"]))


def test_experiment_manifest_keeps_intensity_quantile_execution_provenance(tmp_path):
    config, prepared = tmp_path / "config_resolved.yaml", tmp_path / "observations.npz"
    config.write_text("training: {}\n", encoding="utf-8")
    prepared.write_bytes(b"prepared")
    normalization = {"method": "trimmed_mean", "scale": 2.0, "quantile_execution": "cpu_exact_fallback_for_large_tensor", "input_numel": 123}
    manifest = write_experiment_manifest(
        tmp_path,
        route="test",
        subject_id="CYJ",
        repo_root=Path(__file__).resolve().parents[2],
        config_resolved=config,
        prepared_inputs=[prepared],
        protocol={"tr_ms": 3.2, "vps": 32},
        stack_group_counts={"stack_0": 1},
        seed=1,
        command="test",
        intensity_normalization=normalization,
    )
    saved = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["intensity_normalization"] == normalization == saved["intensity_normalization"]
