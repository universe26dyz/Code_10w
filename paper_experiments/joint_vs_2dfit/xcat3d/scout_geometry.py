"""Local-only XCAT scout geometry discovery, LV-mask audit, and crop proposal."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from ..mrxcat.inspect_case import parse_vti_header
from ..simulation.build_geometry import apply_global_rigid_alignment, select_modal_geometry_templates
from ..simulation.render_observations import midpoint_footprint_world_extrema
from .canonicalize import canonicalize_volume
from .prepare_xcat_parameters import GEOMETRY_TEMPLATE, derive_raw_contract, validate_geometry_spec
from .validate_quantitative_gt import lv_pca_qc


STATIC_FIELDS = {"ap_filename": "AP motion", "dia_filename": "diaphragm motion", "organ_file": "anatomy", "heart_base": "heart anatomy", "heart_curve_file": "heart curve"}
SCOUT_BLOCK = {"pixel_width": 0.1, "slice_width": 0.1, "subvoxel_index": 2, "array_size": 480, "startslice": 950, "endslice": 1350}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path) -> dict[str, str] | None:
    try:
        root = Path(subprocess.check_output(["git", "-C", str(path), "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL).strip())
        return {"repository": str(root), "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()}
    except subprocess.CalledProcessError:
        return None


def _find_suffix(roots: tuple[Path, ...], suffix: str) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        if root.is_dir():
            found.update(path.resolve() for path in root.rglob(suffix) if path.is_file())
    return sorted(found)


def inspect_xcat_scout_context(output_root: str | Path, *, search_roots: tuple[str | Path, ...] = ("/home/universe/SVR", "/home/universe/SVR/multimap_postprogramming")) -> dict:
    """Read local evidence only; this function never invokes XCAT."""
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite scout context {root}")
    roots = tuple(Path(item) for item in search_roots)
    vti_matches = _find_suffix(roots, "Image_01.vti")
    vti_matches = [path for path in vti_matches if str(path).endswith("inpuData/Background/XCAT_vti/Image_01.vti")]
    if len(vti_matches) == 1:
        path = vti_matches[0]
        info = parse_vti_header(path.read_bytes()[:65536].decode("utf-8", errors="ignore"), str(path))
        # The MRXCAT VTI reader documents metre source units; report the
        # physical metadata in mm without importing it into the XCAT frame.
        lower = np.asarray(info.origin) * 1000.0; spacing = np.asarray(info.spacing) * 1000.0
        upper = lower + (np.asarray(info.shape) - 1) * spacing
        mrxcat: dict[str, Any] = {"status": "VERIFIED_LOCAL", "path": str(path), "sha256": _sha256(path), "shape": list(info.shape), "spacing_mm": spacing.tolist(), "origin_mm": lower.tolist(), "direction": list(info.direction), "lower_mm": lower.tolist(), "upper_mm": upper.tolist(), "fov_mm": (upper - lower).tolist(), "git": _git(path.parent)}
    elif len(vti_matches) == 0:
        mrxcat = {"status": "UNAVAILABLE_LOCALLY", "historical_reported_fov_mm": [460.0, 460.0, 144.0]}
    else:
        mrxcat = {"status": "AMBIGUOUS_MULTIPLE_LOCAL_MATCHES", "paths": [str(path) for path in vti_matches]}
    names = set(STATIC_FIELDS) | {"README", "VERSION", "version.txt"}
    candidates = [path for base in roots if base.is_dir() for path in base.rglob("*") if path.is_file() and path.name in names]
    static = {}
    template = json.loads(GEOMETRY_TEMPLATE.read_text())
    for field, role in STATIC_FIELDS.items():
        basename = template["xcat_geometry_block"][field]
        matches = sorted({path.resolve() for path in candidates if path.name == basename})
        static[field] = {"role": role, "basename": basename, "status": "FOUND" if len(matches) == 1 else ("MISSING" if not matches else "AMBIGUOUS"), "matches": [{"path": str(path), "sha256": _sha256(path)} for path in matches]}
    version_evidence = [str(path) for path in candidates if path.name in {"VERSION", "version.txt"} or "readme" in path.name.lower()]
    prepared_candidates = sorted({path.parents[2] for base in roots if base.is_dir() for path in base.rglob("manifest.json") if len(path.parents) >= 3 and path.parent.name == "sax" and path.parent.parent.name in {"CYJ", "DYZ", "HHZ", "HJL"}})
    prepared_status: Any = "REQUIRED_EXTERNAL_PATH"
    for candidate in prepared_candidates:
        try:
            templates = select_modal_geometry_templates(candidate)
            prepared_status = {"status": "RESOLVED_LOCAL", "path": str(candidate), "template_subject": templates["sax"]["source_subject"]}
            break
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
            continue
    repo = Path(__file__).resolve().parents[3]
    contract = derive_raw_contract({**template, "xcat_version_build_compatibility": "SCOUT_UNRESOLVED", "xcat_geometry_block": {**template["xcat_geometry_block"], **SCOUT_BLOCK}})
    unresolved = ["exact licensed XCAT version/build"] + ([] if isinstance(prepared_status, dict) else ["prepared_root"])
    report = {"code_10w": _git(repo), "tracked_worktree_clean": not bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()), "mrxcat": mrxcat, "xcat_version_build": {"status": "UNRESOLVED_EXTERNAL_INFO", "metadata_candidates": version_evidence}, "static_inputs": static, "prepared_root": prepared_status, "geometry_template": {"path": str(GEOMETRY_TEMPLATE), "sha256": _sha256(GEOMETRY_TEMPLATE)}, "scout_candidate": {**SCOUT_BLOCK, "raw_shape_zyx": list(contract["raw_shape"]), "nominal_fov_mm_xyz": [480.0, 480.0, 401.0]}, "unresolved_items": unresolved}
    root.mkdir(parents=True)
    (root / "xcat_scout_context.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (root / "XCAT_SCOUT_CONTEXT.md").write_text("# XCAT scout context\n\nLocal-only discovery; no XCAT executable was run.\n\n```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")
    return report


def make_xcat_scout_geometry(output_root: str | Path, *, xcat_version_build: str | None = None) -> dict:
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite scout geometry directory {root}")
    geometry = json.loads(GEOMETRY_TEMPLATE.read_text())
    geometry["xcat_geometry_block"].update(SCOUT_BLOCK)
    ready = bool(xcat_version_build and xcat_version_build.strip())
    geometry["xcat_version_build_compatibility"] = xcat_version_build.strip() if ready else "UNRESOLVED_EXTERNAL_INFO"
    contract = derive_raw_contract(geometry)
    root.mkdir(parents=True)
    name = "cardiac_roi_geometry.scout_v1.json" if ready else "cardiac_roi_geometry.scout_v1.candidate.json"
    path = root / name
    path.write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n")
    if ready:
        validate_geometry_spec(geometry, formal=True)
    report = {"scout_only": True, "ready_to_freeze": ready, "geometry_path": str(path), "geometry_sha256": _sha256(path), "source_template": str(GEOMETRY_TEMPLATE), "source_template_sha256": _sha256(GEOMETRY_TEMPLATE), "changed_fields": {"xcat_version_build_compatibility": geometry["xcat_version_build_compatibility"], **SCOUT_BLOCK}, "raw_shape_zyx": list(contract["raw_shape"]), "nominal_fov_mm_xyz": [480.0, 480.0, 401.0], "statement": "scout-only; not formal Phase-1.2 geometry"}
    (root / "scout_geometry_design.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (root / "SCOUT_RUN_INSTRUCTIONS.md").write_text("# External scout boundary\n\nRUN ONLY `lv_mask_phase1_2.par` in licensed XCAT. Do not run T1/T2/PD. Return exact XCAT version/build, scout geometry path, LV-mask .par path, raw LV-mask path/SHA256, log, and static-input paths/SHA256. Scout data is never formal Phase-1.2 data.\n")
    return report


def _load_mask(path: Path, *, dtype: str, shape: tuple[int, int, int]) -> np.ndarray:
    data = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.fromfile(path, dtype=np.dtype(dtype))
    if data.size != int(np.prod(shape)):
        raise ValueError(f"scout LV-mask element count {data.size} does not equal expected {int(np.prod(shape))}")
    return np.asarray(data).reshape(shape)


def analyze_xcat_scout(geometry_path: str | Path, lv_mask_raw: str | Path, *, dtype: str, prepared_root: str | Path, output_root: str | Path) -> dict:
    geometry = json.loads(Path(geometry_path).read_text()); validate_geometry_spec(geometry, formal=True)
    contract = derive_raw_contract(geometry); raw_path = Path(lv_mask_raw)
    raw = _load_mask(raw_path, dtype=dtype, shape=contract["raw_shape"])
    values = np.unique(raw)
    if not np.isfinite(raw).all() or values.size > 2 or not np.all(np.isclose(values[:, None], [0.0, 1.0], atol=1e-5).any(axis=1)):
        raise ValueError(f"scout LV mask is not safely binary; unique values: {values[:20].tolist()}")
    binary = raw > 0.5
    if not binary.any():
        raise ValueError("scout LV mask is empty")
    canonical, canonical_geometry = canonicalize_volume(raw, shape=contract["raw_shape"], spacing_mm=contract["raw_spacing_mm"], origin_mm=contract["origin_mm_xyz"], axis_order="zyx", world_convention="LPS", interpolation="nearest")
    mask = canonical > 0.5; raw_box = np.stack((np.argwhere(binary).min(0), np.argwhere(binary).max(0)))
    box = np.stack((np.argwhere(mask).min(0), np.argwhere(mask).max(0)))
    pca = lv_pca_qc(mask, canonical_geometry.origin_mm, canonical_geometry.spacing_mm, minimum_voxel_count=int(geometry["minimum_lv_voxel_count"]), lambda1_lambda2_min=float(geometry["pca_lambda1_lambda2_min"]))
    templates = select_modal_geometry_templates(prepared_root); aligned, alignment = apply_global_rigid_alignment(templates, np.asarray(pca["lv_centroid_mm"]), np.asarray(pca["lv_long_axis"]))
    footprints = {stack: [midpoint_footprint_world_extrema(group, 6, 9) for group in groups] for stack, groups in aligned.items()}
    all_min = np.min([item["world_min_mm"] for values_ in footprints.values() for item in values_], axis=0); all_max = np.max([item["world_max_mm"] for values_ in footprints.values() for item in values_], axis=0)
    lower, upper = box[0] * 1.0, (np.asarray(mask.shape) - 1 - box[1]) * 1.0
    report = {"scout_only": True, "geometry_sha256": _sha256(Path(geometry_path)), "lv_mask": {"path": str(raw_path), "sha256": _sha256(raw_path), "dtype": dtype, "unique_values": values.tolist(), "raw_bbox_zyx": raw_box.tolist(), "canonical_bbox_xyz": box.tolist(), "absolute_xcat_slice_min_max": [int(geometry["xcat_geometry_block"]["startslice"] + raw_box[0, 0]), int(geometry["xcat_geometry_block"]["startslice"] + raw_box[1, 0])], "boundary_distance_mm_xyz": {"lower": lower.tolist(), "upper": upper.tolist()}}, "pca": pca, "template_subject": templates["sax"]["source_subject"], "template_manifest_sha256": {stack: value["source_manifest_sha256"] for stack, value in templates.items()}, "alignment": alignment, "acquisition_footprint_world_min_mm": all_min.tolist(), "acquisition_footprint_world_max_mm": all_max.tolist(), "per_stack_footprints": footprints}
    root = Path(output_root)
    if root.exists(): raise FileExistsError(f"refusing to overwrite scout analysis {root}")
    root.mkdir(parents=True); (root / "xcat_scout_lv_qc.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def propose_final_xcat_geometry(scout_qc_path: str | Path, scout_geometry_path: str | Path, *, safety_margin_mm: float, output_root: str | Path) -> dict:
    scout = json.loads(Path(scout_qc_path).read_text()); geometry = json.loads(Path(scout_geometry_path).read_text())
    contract = derive_raw_contract(geometry); lv_box = np.asarray(scout["lv_mask"]["canonical_bbox_xyz"], float)
    lv_world = np.stack((np.asarray(contract["origin_mm_xyz"]) + lv_box[0], np.asarray(contract["origin_mm_xyz"]) + lv_box[1]))
    required = np.stack((np.minimum(lv_world[0], scout["acquisition_footprint_world_min_mm"]), np.maximum(lv_world[1], scout["acquisition_footprint_world_max_mm"])))
    expanded = required + np.array([[-safety_margin_mm] * 3, [safety_margin_mm] * 3])
    minimum_n = int(np.ceil(2 * max(abs(expanded[0, 0]), abs(expanded[1, 0]), abs(expanded[0, 1]), abs(expanded[1, 1])) + 1))
    recommended_n = max(480, int(np.ceil(minimum_n / 16.0) * 16))
    z_raw = (expanded[:, 2] - contract["origin_mm_xyz"][2]) / contract["canonical_spacing_mm_xyz"][2]
    start = int(geometry["xcat_geometry_block"]["startslice"] + np.floor(z_raw[0])); end = int(geometry["xcat_geometry_block"]["startslice"] + np.ceil(z_raw[1]))
    candidate = json.loads(GEOMETRY_TEMPLATE.read_text()); candidate["xcat_version_build_compatibility"] = geometry["xcat_version_build_compatibility"]; candidate["xcat_geometry_block"].update({"array_size": recommended_n, "startslice": start, "endslice": end})
    validate_geometry_spec(candidate, formal=True); final_contract = derive_raw_contract(candidate)
    final_min = np.asarray(final_contract["origin_mm_xyz"]); final_max = final_min + (np.asarray(final_contract["canonical_shape_xyz"]) - 1) * np.asarray(final_contract["canonical_spacing_mm_xyz"])
    absolute_needed = geometry["xcat_geometry_block"]["startslice"] + z_raw
    xy_gate = bool(np.all(expanded[0, :2] >= final_min[:2]) and np.all(expanded[1, :2] <= final_max[:2]))
    z_gate = bool(start <= np.floor(absolute_needed[0]) and end >= np.ceil(absolute_needed[1]))
    gate = xy_gate and z_gate
    root = Path(output_root)
    if root.exists(): raise FileExistsError(f"refusing to overwrite final proposal {root}")
    root.mkdir(parents=True); candidate_path = root / "cardiac_roi_geometry.final_candidate.json"; candidate_path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    xy_margin = float(np.minimum(expanded[0, :2] - final_min[:2], final_max[:2] - expanded[1, :2]).min())
    z_margin = float(min(np.floor(absolute_needed[0]) - start, end - np.ceil(absolute_needed[1])))
    report = {"scout_only": True, "scout_geometry_sha256": _sha256(Path(scout_geometry_path)), "scout_lv_raw_sha256": scout["lv_mask"]["sha256"], "template_subject": scout["template_subject"], "template_manifest_sha256": scout["template_manifest_sha256"], "safety_margin_mm": safety_margin_mm, "minimum_required_array_size": minimum_n, "recommended_array_size": recommended_n, "minimum_crop_world_mm": required.tolist(), "safety_expanded_world_mm": expanded.tolist(), "required_absolute_slice_min_max": [int(np.floor(absolute_needed[0])), int(np.ceil(absolute_needed[1]))], "recommended_startslice": start, "recommended_endslice": end, "final_candidate": str(candidate_path), "final_candidate_sha256": _sha256(candidate_path), "proposed_final_geometry_gate": gate, "lv_inside_final_crop": z_gate, "acquisition_footprint_inside_final_crop": xy_gate and z_gate, "minimum_final_margin_mm": min(xy_margin, z_margin)}
    (root / "final_geometry_proposal_qc.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
