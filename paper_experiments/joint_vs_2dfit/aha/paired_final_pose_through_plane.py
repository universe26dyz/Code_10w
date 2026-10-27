"""Render qualitative paired through-plane QC with the final-pose common mask.

The quantitative AHA/ROI result remains on native final-pose planes.  This
renderer samples 3-D volumes only for a longitudinal continuity view and
applies the identical paired native support to native, Proposed, and Fit-first.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from ..phase2_clean.visualization import render_through_plane_continuity
from .cyj11_adapter import load_cyj11_native_sax
from .cyj11_myocardial_roi_metrics import ROI_NAMES, _load_scmr_masks
from .cyj11_runner import final_pose_sax_plane, load_cyj11_final_pose_bundles


SCHEMA = "paired_final_pose_through_plane_qc/v1"
SUPPORT_EQUATION = "selected manual anatomical ROI AND finite(reference) AND proposed-final-pose-valid AND fit-first-final-pose-valid AND finite(proposed, fit-first)"


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _native_stack_affine(native: dict[str, np.ndarray]) -> np.ndarray:
    affine_lps = np.asarray(native["affine_lps_rc"], float)
    if affine_lps.ndim != 3 or affine_lps.shape[0] < 2 or affine_lps.shape[1:] != (4, 4):
        raise ValueError("through-plane QC requires at least two affine-defined native SAX groups")
    affine = native_ras_affine(affine_lps[0])
    origins = np.asarray([native_ras_affine(item)[:3, 3] for item in affine_lps], float)
    steps = np.diff(origins, axis=0)
    if not np.allclose(steps, steps[0], atol=1e-4, rtol=0):
        raise ValueError("through-plane QC requires a regular native SAX stack; use per-plane figures otherwise")
    affine[:3, 2] = steps[0]
    return affine


def _support_stack(*, native: dict[str, np.ndarray], bundles: dict[str, object], masks: dict[str, np.ndarray],
                   parameter: str, roi_name: str) -> np.ndarray:
    groups = np.asarray(native["group_idx"], int)
    if roi_name not in masks:
        raise ValueError(f"unknown ROI name: {roi_name}")
    planes: list[np.ndarray] = []
    for index, group in enumerate(groups):
        plane = final_pose_sax_plane(native=native, group=int(group), parameter=parameter, **bundles)
        support = (np.asarray(masks[roi_name][index], bool) & np.isfinite(plane["reference"])
                   & np.asarray(plane["proposed_valid"], bool) & np.asarray(plane["fit_first_valid"], bool)
                   & np.isfinite(plane["proposed"]) & np.isfinite(plane["fit_first"]))
        planes.append(support)
    return np.moveaxis(np.stack(planes), 0, 2)


def build_paired_final_pose_through_plane(*, subject: str, case: str, native_sax_root: str | Path,
                                          prepared_sax: str | Path, roi_bundle: str | Path,
                                          proposed_bundle: str | Path, fit_first_bundle: str | Path,
                                          proposed_t1: str | Path, proposed_t2: str | Path,
                                          fit_first_t1: str | Path, fit_first_t2: str | Path,
                                          output: str | Path, roi_name: str = "myocardium_core_legacy") -> Path:
    """Write masked qualitative through-plane figures and auditable support masks."""
    import nibabel as nib

    native_sax_root, prepared_sax, roi_bundle, output = (Path(value) for value in (native_sax_root, prepared_sax, roi_bundle, output))
    proposed_bundle, fit_first_bundle = Path(proposed_bundle), Path(fit_first_bundle)
    volume_paths = {"t1": (Path(proposed_t1), Path(fit_first_t1)), "t2": (Path(proposed_t2), Path(fit_first_t2))}
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    if roi_name not in ROI_NAMES:
        raise ValueError(f"roi_name must be one of {ROI_NAMES}")
    native = load_cyj11_native_sax(native_sax_root, prepared_sax)
    bundles = load_cyj11_final_pose_bundles(proposed_bundle, fit_first_bundle, subject=subject, case=case)
    groups = np.asarray(native["group_idx"], int)
    masks = _load_scmr_masks(roi_bundle, expected_groups=groups, expected_shape=tuple(map(int, native["t1_ms"].shape[1:])))
    native_affine = _native_stack_affine(native)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        outputs = {}
        reference_group = int(len(groups) // 2)
        for parameter, (proposed_path, fit_path) in volume_paths.items():
            if not proposed_path.is_file() or not fit_path.is_file():
                raise FileNotFoundError(f"{parameter}: Proposed/Fit-first volume is missing")
            proposed_image, fit_image = nib.load(str(proposed_path)), nib.load(str(fit_path))
            support = _support_stack(native=native, bundles=bundles, masks=masks, parameter=parameter, roi_name=roi_name)
            np.savez_compressed(staging / f"{parameter}_native_common_support.npz", group_idx=groups, support=np.moveaxis(support, 2, 0))
            figure = render_through_plane_continuity(
                parameter, np.moveaxis(np.asarray(native[f"{parameter}_ms"], float), 0, 2), native_affine,
                np.asarray(proposed_image.get_fdata(), float), np.asarray(proposed_image.affine, float),
                np.asarray(fit_image.get_fdata(), float), np.asarray(fit_image.affine, float),
                staging / "figures" / f"{parameter}_through_plane_common_mask.png",
                reference_group=reference_group,
                common_support=support,
                metadata={"subject": subject, "case": case, "parameter": parameter,
                          "purpose": "qualitative through-plane continuity QC; final-pose native planes remain the quantitative authority",
                          "support_equation": SUPPORT_EQUATION, "roi_name": roi_name},
            )
            outputs[parameter] = figure
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "roi_name": roi_name,
            "support_equation": SUPPORT_EQUATION, "prepared_sax": _record(prepared_sax),
            "native_sax": {name: _record(native_sax_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "proposed_final_pose_bundle": _record(proposed_bundle / "manifest.json"),
            "fit_first_final_pose_bundle": _record(fit_first_bundle / "manifest.json"),
            "volumes": {parameter: {"proposed": _record(paths[0]), "fit_first": _record(paths[1])} for parameter, paths in volume_paths.items()},
            "outputs": outputs, "quantitative_authority": "paired final-pose AHA/ROI native-plane bundle",
            "qualitative_only": True, "no_training": True, "no_reconstruction": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "case", "native-sax-root", "prepared-sax", "roi-bundle", "proposed-bundle", "fit-first-bundle", "proposed-t1", "proposed-t2", "fit-first-t1", "fit-first-t2", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--roi-name", choices=ROI_NAMES, default="myocardium_core_legacy")
    args = vars(parser.parse_args())
    print(build_paired_final_pose_through_plane(
        subject=args["subject"], case=args["case"], native_sax_root=args["native_sax_root"], prepared_sax=args["prepared_sax"],
        roi_bundle=args["roi_bundle"], proposed_bundle=args["proposed_bundle"], fit_first_bundle=args["fit_first_bundle"],
        proposed_t1=args["proposed_t1"], proposed_t2=args["proposed_t2"], fit_first_t1=args["fit_first_t1"], fit_first_t2=args["fit_first_t2"],
        output=args["output"], roi_name=args["roi_name"],
    ))


if __name__ == "__main__":
    main()
