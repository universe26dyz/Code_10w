"""Immutable complete-hash closure for a completed Phase-1.2 dataset."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from ..simulation.write_observations_npz import validate_observations_file


REQUIRED_ARTIFACTS = (
    "gt_maps.npz", "geometry_qc.json", "geometry_alignment_qc.json",
    "parameter_contract_qc.json", "quantitative_tuple_qc.json", "observation_coverage_qc.json",
    "clean_signal_field.npz", "timing_selection.json", "synthetic_source_manifest.json",
    "sax/observations.npz", "2ch/observations.npz", "4ch/observations.npz",
)
REQUIRED_RAW_XCAT_VOLUMES = ("t1_ms", "t2_ms", "pd_m0", "myocardium_mask")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a complete SHA256")
    return value


def _source_hashes(source: dict) -> tuple[dict[str, str], str, str, dict[str, str], dict[str, str]]:
    xcat = source.get("xcat", {})
    raw = xcat.get("raw_volumes", {})
    if set(raw) != set(REQUIRED_RAW_XCAT_VOLUMES):
        raise ValueError("source raw XCAT set must contain T1, T2, PD and LV mask exactly")
    raw_hashes = {name: _require_sha(raw[name].get("sha256"), f"raw XCAT {name}") for name in REQUIRED_RAW_XCAT_VOLUMES}
    phantom = _require_sha(xcat.get("phantom_spec", {}).get("sha256"), "source cardiac phantom spec")
    geometry = _require_sha(xcat.get("geometry_spec", {}).get("sha256"), "source cardiac geometry spec")
    parameters = xcat.get("parameter_files", {})
    if set(parameters) != {"t1_phase1_2.par", "t2_phase1_2.par", "pd_phase1_2.par", "lv_mask_phase1_2.par"}:
        raise ValueError("source parameter files are incomplete")
    parameter_hashes = {name: _require_sha(record.get("sha256"), f"source parameter {name}") for name, record in parameters.items()}
    static = xcat.get("static_inputs", [])
    if not isinstance(static, list) or not static:
        raise ValueError("source static XCAT input provenance is incomplete")
    static_hashes = {str(item.get("role")): _require_sha(item.get("sha256"), "source static XCAT input") for item in static}
    if len(static_hashes) != len(static) or any(not role or role == "None" for role in static_hashes):
        raise ValueError("source static XCAT input roles must be unique and non-empty")
    return raw_hashes, phantom, geometry, parameter_hashes, static_hashes


def finalize_phase1_2(output_root: str | Path) -> Path:
    root = Path(output_root)
    for item in REQUIRED_ARTIFACTS:
        if not (root / item).is_file():
            raise FileNotFoundError(f"required Phase-1.2 artifact is missing: {item}")
    source, qc, alignment = (json.loads((root / name).read_text()) for name in ("synthetic_source_manifest.json", "geometry_qc.json", "geometry_alignment_qc.json"))
    if source.get("scientific_gt_geometry_gate") is not True or qc.get("scientific_gt_geometry_gate") is not True:
        raise ValueError("Phase-1.2 GT geometry gate is not passing")
    if source.get("lv_boundary_margin_qc", {}).get("gate") is not True:
        raise ValueError("Phase-1.2 LV boundary-margin gate is not passing")
    if source.get("external_reference_provenance", {}).get("reference_commit") != "97558db05f4e344865f472de974b9f4d042a1a15":
        raise ValueError("Phase-1.2 source is not anchored to the required Hanson commit")
    if alignment.get("lv_alignment_gate") is not True:
        raise ValueError("Phase-1.2 LV alignment gate is not passing")
    if alignment.get("pca_stability_gate") is not True or not alignment.get("template_subject"):
        raise ValueError("Phase-1.2 PCA/template geometry gate is not passing")
    if source.get("template_geometry", {}).get("template_subject") != alignment.get("template_subject") or set(source.get("template_geometry", {}).get("manifest_sha256", {})) != {"sax", "2ch", "4ch"}:
        raise ValueError("final source manifest does not prove one coherent template subject")
    coverage = json.loads((root / "observation_coverage_qc.json").read_text())
    if coverage.get("observation_coverage_gate") is not True or any(item.get("gate") is not True for item in coverage.get("stacks", {}).values()):
        raise ValueError("Phase-1.2 observation footprint coverage gate is not passing")
    if source.get("code_provenance", {}).get("tracked_worktree_clean") is not True:
        raise ValueError("formal finalization requires a clean tracked Code_10w worktree")
    maps, signal = np.load(root / "gt_maps.npz"), np.load(root / "clean_signal_field.npz")
    myo, support = maps["myocardium_mask"], maps["signal_support_mask"]
    if not np.array_equal(maps["spacing_mm"], np.ones(3)) or str(maps["axis_order"]) != "xyz" or str(maps["world_convention"]) != "LPS":
        raise ValueError("formal GT is not canonical xyz/LPS/1-mm")
    if not (np.all(maps["t1_ms"][myo] == 1410.0) and np.all(maps["t2_ms"][myo] == 50.0) and np.all(maps["b1"][myo] == 1.0)):
        raise ValueError("frozen LV quantitative values failed")
    if not np.array_equal(signal["signals"][~support], np.zeros((int((~support).sum()), 10), dtype=np.float32)):
        raise ValueError("signal background is not exactly zero")
    if signal["signals"].shape != maps["t1_ms"].shape + (10,) or not np.array_equal(signal["origin_mm"], maps["origin_mm"]) or not np.array_equal(signal["spacing_mm"], maps["spacing_mm"]) or not np.array_equal(signal["signal_support_mask"], support) or not np.isfinite(signal["signals"][support]).all():
        raise ValueError("signal semantic contract failed")
    integration = source.get("spatial_integration", {})
    if integration.get("inplane_samples_row_col") != [6, 6] or integration.get("thickness_samples") != 9:
        raise ValueError("formal renderer is not pinned to 6x6x9")
    actual_centers, source_centers = [], []
    sax_normal = None
    for stack in ("sax", "2ch", "4ch"):
        validate_observations_file(root / stack / "observations.npz")
        obs = np.load(root / stack / "observations.npz", allow_pickle=False)
        for group in range(int(obs["group_idx"].max()) + 1):
            index = np.flatnonzero(obs["group_idx"] == group)[0]
            affine = np.asarray(obs["affine_lps_rc"][index], float)
            rows, cols = obs["images"].shape[1:]
            actual_centers.append(affine[:3, 3] + affine[:3, 0] * (rows - 1) / 2 + affine[:3, 1] * (cols - 1) / 2)
            source_centers.append(np.asarray(alignment["source_group_centers_mm"][stack][group], float))
            if stack == "sax" and group == 0:
                sax_normal = np.cross(affine[:3, 0], affine[:3, 1]); sax_normal /= np.linalg.norm(sax_normal)
    long_axis = np.asarray(alignment["lv_long_axis"], float); long_axis /= np.linalg.norm(long_axis)
    if sax_normal is None or abs(float(np.dot(sax_normal, long_axis))) < 0.999:
        raise ValueError("actual SAX observation affine does not align with LV long axis")
    max_error = max((abs(np.linalg.norm(source_centers[i] - source_centers[j]) - np.linalg.norm(actual_centers[i] - actual_centers[j])) for i in range(len(actual_centers)) for j in range(i + 1, len(actual_centers))), default=0.0)
    if max_error > 1e-6:
        raise ValueError("actual observation affines do not preserve pairwise stack-center distances")
    raw_hashes, phantom_spec_sha256, geometry_spec_sha256, parameter_hashes, static_hashes = _source_hashes(source)
    final = root / "phase1_2_final_manifest.json"
    if final.exists():
        raise FileExistsError(f"refusing to overwrite {final}")
    final.write_text(json.dumps({
        "phase": "1.2", "formal_synthetic_ready": True,
        "artifact_sha256": {item: _sha256(root / item) for item in REQUIRED_ARTIFACTS},
        "raw_xcat_volume_sha256": raw_hashes,
        "cardiac_roi_phantom_spec_sha256": phantom_spec_sha256,
        "cardiac_roi_geometry_spec_sha256": geometry_spec_sha256,
        "parameter_file_sha256": parameter_hashes,
        "static_xcat_input_sha256": static_hashes,
    }, indent=2, sort_keys=True) + "\n")
    return final


def assert_final_manifest_ready(root: str | Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "phase1_2_final_manifest.json").read_text())
    if manifest.get("phase") != "1.2" or manifest.get("formal_synthetic_ready") is not True:
        raise ValueError("Phase-1.2 formal_synthetic_ready is not true")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != set(REQUIRED_ARTIFACTS):
        raise ValueError("artifact_sha256 must contain exactly every required Phase-1.2 artifact")
    raw_hashes = manifest.get("raw_xcat_volume_sha256")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(REQUIRED_RAW_XCAT_VOLUMES):
        raise ValueError("raw_xcat_volume_sha256 must contain T1, T2, PD and LV mask exactly")
    for item in REQUIRED_ARTIFACTS:
        digest = _require_sha(artifact_hashes[item], f"artifact {item}")
        if not (root / item).is_file() or _sha256(root / item) != digest:
            raise ValueError(f"Phase-1.2 final artifact SHA mismatch: {item}")
    source = json.loads((root / "synthetic_source_manifest.json").read_text())
    source_raw, source_phantom, source_geometry, source_parameters, source_static = _source_hashes(source)
    if raw_hashes != source_raw or manifest.get("cardiac_roi_phantom_spec_sha256") != source_phantom or manifest.get("cardiac_roi_geometry_spec_sha256") != source_geometry or manifest.get("parameter_file_sha256") != source_parameters or manifest.get("static_xcat_input_sha256") != source_static:
        raise ValueError("final provenance disagrees with hashed synthetic source manifest")
    for name in REQUIRED_RAW_XCAT_VOLUMES:
        _require_sha(raw_hashes[name], f"raw XCAT {name}")
    _require_sha(manifest.get("cardiac_roi_phantom_spec_sha256"), "cardiac phantom spec")
    _require_sha(manifest.get("cardiac_roi_geometry_spec_sha256"), "cardiac geometry spec")
    return manifest
