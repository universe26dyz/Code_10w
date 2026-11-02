"""Export small, read-only native/final-pose parity-audit records.

Run ``current`` on the ISMRM server and ``historical`` beside the archived
SCMR outputs.  The resulting JSON files are intentionally small enough to
transfer independently of images and reconstructions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_myocardial_roi_metrics import _load_scmr_masks
from .cyj11_runner import load_cyj11_run_inputs, native_sax_plane


CURRENT_SCHEMA = "cyj11_parity_current_native/v1"
HISTORICAL_SCHEMA = "cyj11_parity_historical_final_pose/v1"
ROI_NAMES = ("myocardium_full", "myocardium_core_legacy", "myocardium_core_1px")


def _mask_evidence(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, bool)
    coordinates = np.argwhere(mask)
    digest = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    return {"voxels": int(mask.sum()), "shape": list(mask.shape), "sha256": digest,
            "bbox_rc": None if not coordinates.size else [int(coordinates[:, 0].min()), int(coordinates[:, 0].max()),
                                                           int(coordinates[:, 1].min()), int(coordinates[:, 1].max())]}


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = np.asarray(values, float)[np.asarray(mask, bool)]
    return None if not selected.size else float(selected.mean())


def write_current_parity_audit(*, native: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray],
                               endpoint_planes: Mapping[tuple[int, str], Mapping[str, np.ndarray]],
                               subject: str, case: str, plane_metadata: Mapping[int, Mapping[str, Any]],
                               output: str | Path, manifest_updates: Mapping[str, Any] | None = None) -> Path:
    """Write per-plane native-crop ROI/support/map evidence, without changing inputs."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    groups = np.asarray(native["group_idx"], int)
    t1, t2 = np.asarray(native["t1_ms"], float), np.asarray(native["t2_ms"], float)
    if t1.ndim != 3 or t2.shape != t1.shape or t1.shape[0] != len(groups):
        raise ValueError("current parity audit requires equal-shape native T1/T2 stacks")
    if any(np.asarray(masks[name]).shape != t1.shape for name in ROI_NAMES):
        raise ValueError("current parity audit ROI stacks must match native maps")
    records: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group = int(group)
        metadata = plane_metadata.get(group)
        if metadata is None:
            raise ValueError(f"current parity audit lacks ROI metadata for group {group}")
        record: dict[str, Any] = {
            "native_group_idx": group, "manual_roi_slice_index": int(metadata["roi_slice_index"]),
            "native_shape_rc": list(t1[index].shape), "affine_lps_rc": np.asarray(native["affine_lps_rc"][index], float).tolist(),
            "rois": {name: _mask_evidence(masks[name][index]) for name in ROI_NAMES}, "parameters": {},
        }
        for parameter, reference in (("t1", t1[index]), ("t2", t2[index])):
            plane = endpoint_planes[(group, parameter)]
            support = (np.isfinite(reference) & np.isfinite(plane["proposed"]) & np.isfinite(plane["fit_first"])
                       & np.asarray(plane["proposed_valid"], bool) & np.asarray(plane["fit_first_valid"], bool))
            detail: dict[str, Any] = {"common_support_voxels": int(support.sum()), "common_support": _mask_evidence(support)}
            for name in ROI_NAMES:
                selected = support & np.asarray(masks[name][index], bool)
                roi_reference = np.asarray(masks[name][index], bool) & np.isfinite(reference)
                detail[name.replace("myocardium_", "")] = {
                    "support_voxels": int(selected.sum()), "support": _mask_evidence(selected),
                    "reference_roi_mean": _mean(reference, roi_reference),
                    "reference_mean": _mean(reference, selected), "proposed_mean": _mean(plane["proposed"], selected),
                    "fit_first_mean": _mean(plane["fit_first"], selected),
                }
            record["parameters"][parameter] = detail
        records.append(record)
    payload: dict[str, Any] = {"schema": CURRENT_SCHEMA, "subject": subject, "case": case, "planes": records,
        "support_equation": "ROI AND finite(reference) AND proposed-valid AND fit-first-valid AND finite(proposed, fit-first)",
        "purpose": "evidence export only; not a formal metric", "downstream_only": True, "immutable_output": True}
    if manifest_updates:
        payload.update(dict(manifest_updates))
    return write_immutable_json(output, payload)


def build_current_parity_audit(*, analysis_root: str | Path, subject: str, case: str, result_tag: str,
                               output: str | Path) -> Path:
    root = Path(analysis_root)
    metric_manifest_path = root / "metrics" / "central" / subject / case / result_tag / "myocardial_roi" / "manifest.json"
    metric_manifest = json.loads(metric_manifest_path.read_text(encoding="utf-8"))
    if metric_manifest.get("schema") != "cyj11_local_myocardial_roi_metrics/v1":
        raise ValueError("current parity audit requires completed CYJ1.1 myocardial ROI output")
    bundle = Path(str(metric_manifest["scmr_myocardium_bundle"]["archive"]["path"])).parent
    derivation = json.loads((bundle / "roi_derivation_manifest.json").read_text(encoding="utf-8"))
    plane_metadata = {int(item["native_group_idx"]): item for item in derivation["planes"]}
    native, endpoints = load_cyj11_run_inputs(root, subject, case)
    masks = _load_scmr_masks(bundle, expected_groups=np.asarray(native["group_idx"], int),
                             expected_shape=tuple(map(int, native["t1_ms"].shape[1:])))
    planes = {(int(group), parameter): native_sax_plane(native=native, group=int(group), endpoint_images=endpoints,
                                                          parameter=parameter)
              for group in native["group_idx"] for parameter in ("t1", "t2")}
    return write_current_parity_audit(native=native, masks=masks, endpoint_planes=planes, subject=subject, case=case,
        plane_metadata=plane_metadata, output=output, manifest_updates={"result_tag": result_tag,
        "source_myocardial_manifest": {"path": str(metric_manifest_path.resolve()), "sha256": sha256(metric_manifest_path)},
        "roi_bundle": {"path": str(bundle.resolve()), "manifest_sha256": sha256(bundle / "manifest.json")}})


def _nifti_values(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import nibabel as nib
    image = nib.load(str(path))
    values = np.asarray(image.get_fdata(), float).squeeze()
    if values.ndim != 2:
        raise ValueError(f"historical parity audit requires 2-D NIfTI: {path}")
    return values, {"shape": list(image.shape), "affine_ras": np.asarray(image.affine, float).tolist(), "sha256": sha256(path)}


def build_historical_parity_audit(*, historical_eval: str | Path, output: str | Path) -> Path:
    """Export legacy final-pose core/full support evidence on the local SCMR host."""
    root = Path(historical_eval)
    roi_manifest_path, common_path = root / "manifests" / "roi_final_pose_manifest.json", root / "manifests" / "common_mask_manifest.json"
    roi_manifest, common_manifest = (json.loads(path.read_text(encoding="utf-8")) for path in (roi_manifest_path, common_path))
    if roi_manifest.get("status") != "PASS":
        raise ValueError("historical parity audit requires PASS roi_final_pose_manifest.json")
    rois = {(str(x["parameter"]).lower(), int(x["native_slice_id"]), str(x["roi"])): Path(x["path"])
            for x in roi_manifest.get("mappings", [])}
    records: list[dict[str, Any]] = []
    for item in common_manifest.get("slices", []):
        if item.get("stack") != "sax":
            continue
        parameter, native_id = str(item["parameter"]).lower(), int(item["native_slice_id"])
        core_path, full_path = rois.get((parameter, native_id, "myocardium_core")), rois.get((parameter, native_id, "myocardium_full"))
        if core_path is None or full_path is None:
            raise ValueError(f"historical parity audit lacks core/full ROI for {parameter} SAX {native_id}")
        original, geometry = _nifti_values(Path(item["original_ms"]))
        synthetic, _ = _nifti_values(Path(item["synthetic_ms"]))
        common, _ = _nifti_values(Path(item["common_mask"]))
        core, _ = _nifti_values(core_path); full, _ = _nifti_values(full_path)
        if not (original.shape == synthetic.shape == common.shape == core.shape == full.shape):
            raise ValueError(f"historical parity audit array shape mismatch for {parameter} SAX {native_id}")
        support = np.isfinite(original) & np.isfinite(synthetic) & (common > 0)
        details = {}
        for name, roi in (("core_legacy", core > 0), ("full", full > 0)):
            selected = support & roi
            details[name] = {"roi": _mask_evidence(roi), "support_voxels": int(selected.sum()),
                             "support": _mask_evidence(selected),
                             "reference_roi_mean": _mean(original, roi & np.isfinite(original)),
                             "reference_mean": _mean(original, selected),
                             "historical_reprojection_mean": _mean(synthetic, selected)}
        records.append({"parameter": parameter, "native_slice_id": native_id, "output_id": int(item["output_id"]),
                        "final_pose_geometry": geometry, "common_support": _mask_evidence(support), "rois": details})
    payload = {"schema": HISTORICAL_SCHEMA, "subject": roi_manifest.get("subject_id"), "historical_eval": str(root.resolve()),
               "planes": records, "support_equation": "historical common_mask AND finite(original, synthetic) AND final-pose ROI",
               "purpose": "evidence export only; not a formal metric", "immutable_output": True,
               "source_manifests": {"roi_final_pose": {"path": str(roi_manifest_path.resolve()), "sha256": sha256(roi_manifest_path)},
                                    "common_mask": {"path": str(common_path.resolve()), "sha256": sha256(common_path)}}}
    return write_immutable_json(output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("current", "historical"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--subject")
    parser.add_argument("--case")
    parser.add_argument("--result-tag", default="cyj11_scmr_aha_roi_20261026")
    parser.add_argument("--historical-eval")
    args = parser.parse_args()
    if args.mode == "current":
        if not all((args.analysis_root, args.subject, args.case)):
            raise ValueError("current mode requires --analysis-root, --subject, and --case")
        print(build_current_parity_audit(analysis_root=args.analysis_root, subject=args.subject, case=args.case,
              result_tag=args.result_tag, output=args.output))
    else:
        if not args.historical_eval:
            raise ValueError("historical mode requires --historical-eval")
        print(build_historical_parity_audit(historical_eval=args.historical_eval, output=args.output))


if __name__ == "__main__":
    main()
