"""Read-only audit of NeSVoR sampling masks for existing sparse fit-first models."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256
from .evaluation import _world_sample, reproject_volume_to_native_plane, resample_volume_to_grid
from .fit_first import select_verified_native_cache_groups


NORMALIZATION_SCALES = {"t1": 2500.0, "t2": 200.0}
INFERENCE_PARAMETERS = {
    "output_resolution_mm": 1.0,
    "output_psf_factor": 1,
    "n_inference_samples": 128,
    "inference_batch_size": 16000,
    "seed": 20260823,
    "output_intensity_mean": 0,
}
SELF_RESAMPLE_RMSE_TOLERANCE_MS = 1e-3
SELF_RESAMPLE_MAX_ABSOLUTE_TOLERANCE_MS = 1e-2

_EXPORT_CHECKPOINT_MASK = """
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location=\"cpu\")
mask = checkpoint.get(\"mask\")
if mask is None:
    raise RuntimeError(\"NeSVoR checkpoint has no saved mask\")
mask.save_mask(sys.argv[2])
"""

_SAMPLE_CHECKPOINT_WITH_MASK = """
import hashlib
import json
import sys
import torch
import nesvor
import nesvor.inr.models as models_module
import nesvor.inr.sample as sample_module
from nesvor.image import load_mask
from nesvor.inr.models import INR
from nesvor.inr.sample import sample_volume
from nesvor.utils import set_seed

checkpoint_path, sample_mask_path, output_path, runtime_path, requested_device = sys.argv[1:6]
if requested_device == \"-1\":
    device = torch.device(\"cpu\")
elif requested_device.isdigit():
    device = torch.device(\"cuda:\" + requested_device)
    if not torch.cuda.is_available():
        raise RuntimeError(\"requested CUDA device is unavailable\")
else:
    raise ValueError(\"device must be a CUDA index such as 0, or -1 for test/debug CPU\")

def require_device(name, value):
    if value.device != device:
        raise RuntimeError(\"device mismatch: %s is %s, expected %s\" % (name, value.device, device))

def sha256(path):
    digest = hashlib.sha256()
    with open(path, \"rb\") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b\"\"):
            digest.update(block)
    return digest.hexdigest()

set_seed(20260823)
checkpoint = torch.load(checkpoint_path, map_location=device)
args = checkpoint[\"args\"]
args.device = device
model = INR(checkpoint[\"model\"][\"bounding_box\"], args)
model.load_state_dict(checkpoint[\"model\"])
model.to(device)
model.eval()
sample_mask = load_mask(sample_mask_path, device=device)
sample_mask = sample_mask.resample(1.0, None)
require_device(\"model_parameter\", next(model.parameters()))
require_device(\"model_bounding_box\", model.bounding_box)
require_device(\"sample_mask.image\", sample_mask.image)
require_device(\"sample_mask.mask\", sample_mask.mask)
if sample_mask.transformation.device != device:
    raise RuntimeError(\"device mismatch: sample_mask.transformation is %s, expected %s\" % (sample_mask.transformation.device, device))
require_device(\"sample_mask.xyz_masked\", sample_mask.xyz_masked)
sampled = sample_volume(model, sample_mask, 1.0, 16000, 128)
sampled.save(output_path, masked=True)
runtime = {
    \"sampling_backend\": \"nesvor_adjacent_python_direct_sample_volume\",
    \"nesvor_python_executable\": sys.executable,
    \"nesvor_package_path\": nesvor.__file__,
    \"sample_source_path\": sample_module.__file__,
    \"models_source_path\": models_module.__file__,
    \"sample_source_sha256\": sha256(sample_module.__file__) if sample_module.__file__ else None,
    \"models_source_sha256\": sha256(models_module.__file__) if models_module.__file__ else None,
    \"requested_device\": requested_device,
    \"resolved_torch_device\": str(device),
    \"model_device\": str(next(model.parameters()).device),
    \"model_bounding_box_device\": str(model.bounding_box.device),
    \"sample_mask_image_device\": str(sample_mask.image.device),
    \"sample_mask_mask_device\": str(sample_mask.mask.device),
    \"sample_mask_transformation_device\": str(sample_mask.transformation.device),
    \"sample_mask_xyz_masked_device\": str(sample_mask.xyz_masked.device),
    \"inference_parameters\": {\"output_resolution_mm\": 1.0, \"output_psf_factor\": 1, \"psf_resolution_mm\": 1.0, \"n_inference_samples\": 128, \"inference_batch_size\": 16000, \"seed\": 20260823},
}
with open(runtime_path, \"w\") as handle:
    json.dump(runtime, handle, indent=2, sort_keys=True)
"""


def require_new_output(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def _file_record(path: str | Path) -> dict[str, str | None]:
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path) if path.is_file() else None}


def mask_nifti_summary(path: str | Path) -> dict[str, Any]:
    """Report the physical NIfTI produced directly by the checkpoint mask object."""
    import nibabel as nib

    image = nib.load(str(path))
    values = image.get_fdata()
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "shape": list(image.shape),
        "affine": image.affine.tolist(),
        "mask_voxel_count": int(np.count_nonzero(values > 0)),
    }


def checkpoint_mask_export_command(
    nesvor_python: str | Path, checkpoint: str | Path, output_mask: str | Path
) -> list[str]:
    """Use the NeSVoR runtime to unpickle and save its real checkpoint mask."""
    return [str(nesvor_python), "-c", _EXPORT_CHECKPOINT_MASK, str(checkpoint), str(output_mask)]


def resolve_torch_device(device: str) -> str:
    if device == "-1":
        return "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    raise ValueError("device must be a CUDA index such as 0, or -1 for test/debug CPU")


def sample_volume_runtime_command(
    nesvor_python: str | Path,
    model: str | Path,
    sample_mask: str | Path,
    output_volume: str | Path,
    runtime_output: str | Path,
    *,
    device: str,
) -> list[str]:
    """Run the existing NeSVoR sampler with explicit model/mask device checks."""
    resolve_torch_device(device)
    return [
        str(nesvor_python), "-c", _SAMPLE_CHECKPOINT_WITH_MASK,
        str(model), str(sample_mask), str(output_volume), str(runtime_output), str(device),
    ]


def _metric_summary(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    support = np.asarray(support, bool) & np.isfinite(reference) & np.isfinite(prediction)
    if not support.any():
        return {"voxel_count": 0, "rmse": None, "mae": None, "bias": None, "reference_mean_ms": None, "prediction_mean_ms": None}
    reference, prediction = np.asarray(reference, float)[support], np.asarray(prediction, float)[support]
    difference = prediction - reference
    return {
        "voxel_count": int(support.sum()),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "mae": float(np.mean(np.abs(difference))),
        "bias": float(np.mean(difference)),
        "reference_mean_ms": float(np.mean(reference)),
        "prediction_mean_ms": float(np.mean(prediction)),
    }


def mask_support_audit(
    l100_mask: np.ndarray, l100_affine: np.ndarray, sparse_mask: np.ndarray, sparse_affine: np.ndarray
) -> dict[str, Any]:
    """Nearest-neighbour compare sparse mask occupancy on the L100 mask world grid."""
    l100 = np.asarray(l100_mask, bool)
    indices = np.indices(l100.shape, dtype=float).reshape(3, -1)
    world = np.asarray(l100_affine, float) @ np.vstack((indices, np.ones(indices.shape[1])))
    sparse_on_l100, inside = _world_sample(
        np.asarray(sparse_mask, np.uint8), sparse_affine, world.reshape(4, *l100.shape), order=0
    )
    sparse = inside & np.isfinite(sparse_on_l100) & (sparse_on_l100 > .5)
    intersection = l100 & sparse
    return {
        "comparison_grid": "l100_mask_world_grid_nearest",
        "l100_mask_voxel_count": int(l100.sum()),
        "sparse_mask_voxel_count": int(sparse.sum()),
        "sparse_source_mask_voxel_count": int(np.asarray(sparse_mask, bool).sum()),
        "intersection_voxel_count": int(intersection.sum()),
        "sparse_to_l100_coverage_ratio": float(sparse.sum() / l100.sum()) if l100.any() else None,
        "l100_only_voxel_count": int((l100 & ~sparse).sum()),
    }


def heldout_mask_plane_audit(
    reference: np.ndarray,
    default_prediction: np.ndarray,
    fixed_prediction: np.ndarray,
    source_support: np.ndarray,
    default_valid: np.ndarray,
    fixed_valid: np.ndarray,
) -> dict[str, Any]:
    """Audit fixed-mask recovery without changing the held-out formal endpoint."""
    reference = np.asarray(reference, float)
    source = np.asarray(source_support, bool) & np.isfinite(reference)
    default = source & np.asarray(default_valid, bool) & np.isfinite(default_prediction)
    fixed = source & np.asarray(fixed_valid, bool) & np.isfinite(fixed_prediction)
    both = default & fixed
    recovered = fixed & ~default
    return {
        "native_source_support_voxels": int(source.sum()),
        "default_sparse_valid_coverage_voxels": int(default.sum()),
        "fixed_l100_mask_valid_coverage_voxels": int(fixed.sum()),
        "newly_recovered_voxel_count": int(recovered.sum()),
        "both_valid_voxel_count": int(both.sum()),
        "both_valid_default_vs_fixed": _metric_summary(default_prediction, fixed_prediction, both),
        "newly_recovered_fixed_vs_native": _metric_summary(reference, fixed_prediction, recovered),
    }


def audit_manifest(
    *, sparse_manifest: str | Path, r0_native_maps_root: str | Path,
    nesvor_executable: str | Path, checkpoints: Mapping[str, Mapping[str, str | Path]],
    sampling_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_checkpoints": {parameter: {name: _file_record(path) for name, path in roots.items()} for parameter, roots in checkpoints.items()},
        "sparse_manifest": _file_record(sparse_manifest),
        "r0_native_maps_root": str(Path(r0_native_maps_root).resolve()),
        "nesvor_executable": str(nesvor_executable),
        "sampling_backend": "nesvor_adjacent_python_direct_sample_volume",
        "sampling_runtime": dict(sampling_runtime or {}),
        "inference_parameters": INFERENCE_PARAMETERS,
        "normalization_scales": NORMALIZATION_SCALES,
        "no_training": True,
        "no_refitting": True,
        "scientific_change": "sampling_mask_only",
        "git_sha": _git_sha(),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _denormalize(normalized: Path, output: Path, scale: float) -> None:
    import nibabel as nib

    image = nib.load(str(normalized))
    nib.save(nib.Nifti1Image(image.get_fdata(dtype=np.float32) * scale, image.affine, image.header), str(output))


def _heldout_ids(sparse: Mapping[str, Any], stack: str) -> list[int]:
    spec = sparse["stacks"][stack]
    if sparse.get("condition") != "L100":
        return [int(group) for group in spec["heldout_original_group_ids"]]
    with np.load(spec["source_observation"]["path"], allow_pickle=False) as source:
        return [int(group) for group in np.unique(source["group_idx"])]


def _final_output_mask_summary(path: Path, staging: Path, output: Path) -> dict[str, Any]:
    result = mask_nifti_summary(path)
    result["path"] = str((output / path.relative_to(staging)).resolve())
    return result


def _validate_inputs(l100_root: Path, sparse_root: Path, sparse_manifest: Path, r0_cache: Path, nesvor: Path) -> None:
    required = [sparse_manifest, nesvor]
    for root in (l100_root, sparse_root):
        for parameter in NORMALIZATION_SCALES:
            required.append(root / "nesvor" / parameter / "model.pt")
    for parameter in NORMALIZATION_SCALES:
        required.extend((sparse_root / "nesvor" / parameter / "volume_normalized.nii.gz", sparse_root / "nesvor" / parameter / "volume_ms.nii.gz"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("mask audit input is incomplete: " + ", ".join(missing))
    if not r0_cache.is_dir():
        raise FileNotFoundError(f"R0 native cache is unavailable: {r0_cache}")
    from .visualization import validate_fit_first_scale
    for parameter in NORMALIZATION_SCALES:
        validate_fit_first_scale(sparse_root / "nesvor" / parameter, parameter.upper())


def _write_summary_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for parameter, result in report["parameters"].items():
        for metric, value in result["self_resample"].items():
            rows.append({"parameter": parameter, "section": "self_resample", "stack": "", "group_index": "", "metric": metric, "value": value})
        for metric, value in result["mask_support"].items():
            rows.append({"parameter": parameter, "section": "mask_support", "stack": "", "group_index": "", "metric": metric, "value": value})
        for stack, planes in result["heldout_native_planes"].items():
            for plane in planes:
                for metric in ("native_source_support_voxels", "default_sparse_valid_coverage_voxels", "fixed_l100_mask_valid_coverage_voxels", "newly_recovered_voxel_count", "both_valid_voxel_count"):
                    rows.append({"parameter": parameter, "section": "heldout_native", "stack": stack, "group_index": plane["original_group_idx"], "metric": metric, "value": plane[metric]})
                for section in ("both_valid_default_vs_fixed", "newly_recovered_fixed_vs_native"):
                    for metric, value in plane[section].items():
                        rows.append({"parameter": parameter, "section": section, "stack": stack, "group_index": plane["original_group_idx"], "metric": metric, "value": value})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("parameter", "section", "stack", "group_index", "metric", "value"))
        writer.writeheader(); writer.writerows(rows)


def run_mask_audit(
    *, l100_fit_first_root: str | Path, sparse_fit_first_root: str | Path,
    sparse_manifest: str | Path, r0_native_maps_root: str | Path,
    nesvor_executable: str | Path, output: str | Path, device: str,
) -> Path:
    """Run the sampling-only audit; no training, refitting, or output mutation occurs."""
    import nibabel as nib

    l100_root, sparse_root = Path(l100_fit_first_root), Path(sparse_fit_first_root)
    sparse_manifest, r0_cache, nesvor, output = Path(sparse_manifest), Path(r0_native_maps_root), Path(nesvor_executable), require_new_output(output)
    _validate_inputs(l100_root, sparse_root, sparse_manifest, r0_cache, nesvor)
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    if sparse.get("status") != "ELIGIBLE":
        raise ValueError("mask audit requires an ELIGIBLE sparse manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    checkpoints = {parameter: {"sparse": sparse_root / "nesvor" / parameter / "model.pt", "l100": l100_root / "nesvor" / parameter / "model.pt"} for parameter in NORMALIZATION_SCALES}
    nesvor_python = nesvor.with_name("python")
    if not nesvor_python.is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise FileNotFoundError(f"NeSVoR Python adjacent to executable is unavailable: {nesvor_python}")
    try:
        selected = {stack: select_verified_native_cache_groups(r0_cache, sparse_manifest, stack, _heldout_ids(sparse, stack)) for stack in ("2ch", "4ch")}
        report: dict[str, Any] = {"audit": "sparse_fit_first_sampling_mask_only", "parameters": {}}
        manifest = audit_manifest(sparse_manifest=sparse_manifest, r0_native_maps_root=r0_cache, nesvor_executable=nesvor, checkpoints=checkpoints)
        manifest["r0_native_cache_provenance"] = {stack: selected[stack]["provenance"] for stack in selected}
        manifest["sampling_runtime"] = {}
        for parameter, scale in NORMALIZATION_SCALES.items():
            target = staging / parameter; target.mkdir()
            sparse_mask, l100_mask = target / "sparse_model_mask.nii.gz", target / "l100_model_mask.nii.gz"
            _run(checkpoint_mask_export_command(nesvor_python, checkpoints[parameter]["sparse"], sparse_mask))
            _run(checkpoint_mask_export_command(nesvor_python, checkpoints[parameter]["l100"], l100_mask))
            self_normalized = target / "self_mask" / "volume_normalized.nii.gz"; self_normalized.parent.mkdir()
            fixed_normalized = target / "l100_mask" / "volume_normalized.nii.gz"; fixed_normalized.parent.mkdir()
            self_runtime, fixed_runtime = self_normalized.with_name("runtime.json"), fixed_normalized.with_name("runtime.json")
            _run(sample_volume_runtime_command(nesvor_python, checkpoints[parameter]["sparse"], sparse_mask, self_normalized, self_runtime, device=device))
            self_ms = self_normalized.with_name("volume_ms.nii.gz")
            _denormalize(self_normalized, self_ms, scale)
            formal = nib.load(str(sparse_root / "nesvor" / parameter / "volume_ms.nii.gz"))
            self_image = nib.load(str(self_ms))
            self_on_formal, self_inside = resample_volume_to_grid(self_image.get_fdata(), self_image.affine, formal.shape, formal.affine)
            self_metrics = _metric_summary(formal.get_fdata(), self_on_formal, self_inside)
            self_common = self_inside & np.isfinite(formal.get_fdata()) & np.isfinite(self_on_formal)
            self_metrics["finite_common_voxel_count"] = int(self_common.sum())
            self_metrics["max_absolute_difference"] = float(np.max(np.abs(formal.get_fdata()[self_common] - self_on_formal[self_common]))) if self_common.any() else None
            self_metrics["status"] = "PASS" if self_metrics["rmse"] is not None and self_metrics["rmse"] <= SELF_RESAMPLE_RMSE_TOLERANCE_MS and self_metrics["max_absolute_difference"] is not None and self_metrics["max_absolute_difference"] <= SELF_RESAMPLE_MAX_ABSOLUTE_TOLERANCE_MS else "FAIL"
            self_metrics["rmse_tolerance_ms"] = SELF_RESAMPLE_RMSE_TOLERANCE_MS
            self_metrics["max_absolute_tolerance_ms"] = SELF_RESAMPLE_MAX_ABSOLUTE_TOLERANCE_MS
            if self_metrics["status"] != "PASS":
                raise RuntimeError(f"{parameter}: self-resample sanity check failed; fixed-mask result will not be interpreted")
            _run(sample_volume_runtime_command(nesvor_python, checkpoints[parameter]["sparse"], l100_mask, fixed_normalized, fixed_runtime, device=device))
            fixed_ms = fixed_normalized.with_name("volume_ms.nii.gz")
            _denormalize(fixed_normalized, fixed_ms, scale)
            fixed_image = nib.load(str(fixed_ms))
            l100_image, sparse_mask_image = nib.load(str(l100_mask)), nib.load(str(sparse_mask))
            parameter_report: dict[str, Any] = {"self_resample": self_metrics, "mask_support": mask_support_audit(l100_image.get_fdata() > 0, l100_image.affine, sparse_mask_image.get_fdata() > 0, sparse_mask_image.affine), "heldout_native_planes": {}}
            for stack, cache in selected.items():
                planes = []
                for index, group in enumerate(cache["group_idx"]):
                    affine_ras = np.diag((-1.0, -1.0, 1.0, 1.0)) @ np.asarray(cache["affine_lps_rc"][index], float)
                    default, default_valid = reproject_volume_to_native_plane(formal.get_fdata(), formal.affine, affine_ras, cache[f"{parameter}_ms"][index].shape)
                    fixed, fixed_valid = reproject_volume_to_native_plane(fixed_image.get_fdata(), fixed_image.affine, affine_ras, cache[f"{parameter}_ms"][index].shape)
                    plane = heldout_mask_plane_audit(cache[f"{parameter}_ms"][index], default, fixed, cache["source_support"][index], default_valid, fixed_valid)
                    plane["original_group_idx"] = int(group); planes.append(plane)
                parameter_report["heldout_native_planes"][stack] = planes
            report["parameters"][parameter] = parameter_report
            manifest.setdefault("mask_outputs", {})[parameter] = {"sparse": _final_output_mask_summary(sparse_mask, staging, output), "l100": _final_output_mask_summary(l100_mask, staging, output)}
            runtime_records = {"self_mask": json.loads(self_runtime.read_text(encoding="utf-8")), "l100_mask": json.loads(fixed_runtime.read_text(encoding="utf-8"))}
            manifest["sampling_runtime"][parameter] = runtime_records
            for key in ("nesvor_python_executable", "nesvor_package_path", "sample_source_path", "models_source_path", "sample_source_sha256", "models_source_sha256", "requested_device", "resolved_torch_device", "model_device", "model_bounding_box_device", "sample_mask_image_device", "sample_mask_mask_device", "sample_mask_transformation_device", "sample_mask_xyz_masked_device"):
                manifest.setdefault(key, runtime_records["self_mask"][key])
        _write_summary_csv(staging / "audit_summary.csv", report)
        (staging / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("l100-fit-first-root", "sparse-fit-first-root", "sparse-manifest", "r0-native-maps-root", "nesvor-executable", "output", "device"):
        parser.add_argument("--" + name, required=True)
    print(run_mask_audit(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
