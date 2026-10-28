"""Add missing common-native visual outputs without recomputing existing metrics."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import write_immutable_json
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from ..phase2_clean.visualization import native_display_support, render_through_plane_continuity
from .common_native import load_volume_images, sample_common_native_plane
from .common_native_metrics_qc import (PARAMETERS, SUPPORT_EQUATION, _native_stack_affine,
                                       _render_aha_outputs, _volume_arrays)
from .cyj11_adapter import derive_cyj11_historical_mapping, load_cyj11_native_sax
from .cyj11_myocardial_roi_metrics import _load_scmr_masks


def _sax_planes(*, native: dict[str, np.ndarray], images: dict[str, dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {parameter: {} for parameter in PARAMETERS}
    for index, group_value in enumerate(np.asarray(native["group_idx"], int)):
        group = int(group_value)
        affine = native_ras_affine(np.asarray(native["affine_lps_rc"][index], float))
        for parameter in PARAMETERS:
            proposed, proposed_affine, fit_first, fit_first_affine = _volume_arrays(images, parameter)
            result[parameter][group] = sample_common_native_plane(
                reference=np.asarray(native[f"{parameter}_ms"][index], float), native_affine_ras=affine,
                proposed_volume=proposed, proposed_affine_ras=proposed_affine,
                fit_first_volume=fit_first, fit_first_affine_ras=fit_first_affine,
            )
    return result


def _render_through_plane(*, staging: Path, subject: str, case: str, native: dict[str, np.ndarray],
                          images: dict[str, dict[str, Any]], sax_planes: dict[str, dict[int, dict[str, Any]]]) -> None:
    groups = np.asarray(native["group_idx"], int)
    affine = _native_stack_affine(native)
    for parameter in PARAMETERS:
        support = np.moveaxis(np.stack([
            native_display_support(sax_planes[parameter][int(group)]["reference"],
                                   sax_planes[parameter][int(group)]["proposed"],
                                   sax_planes[parameter][int(group)]["fit_first"],
                                   sax_planes[parameter][int(group)]["proposed_valid"],
                                   sax_planes[parameter][int(group)]["fit_first_valid"])
            for group in groups
        ]), 0, 2)
        proposed, proposed_affine, fit_first, fit_first_affine = _volume_arrays(images, parameter)
        native_stack = np.moveaxis(np.asarray(native[f"{parameter}_ms"], float), 0, 2)
        for orientation in ("column", "row"):
            render_through_plane_continuity(
                parameter, native_stack, affine, proposed, proposed_affine, fit_first, fit_first_affine,
                staging / "through_plane" / parameter / orientation / f"{parameter}_through_plane_{orientation}.png",
                reference_group=len(groups) // 2, common_support=support, cut_orientation=orientation,
                metadata={"subject": subject, "case": case, "parameter": parameter,
                          "purpose": "common-native qualitative through-plane continuity QC",
                          "support_equation": SUPPORT_EQUATION},
            )


def build_common_native_visual_backfill(*, subject: str, case: str, native_maps_root: str | Path,
                                        prepared_subject_root: str | Path, roi_bundle: str | Path,
                                        proposed_t1: str | Path, proposed_t2: str | Path,
                                        fit_first_t1: str | Path, fit_first_t2: str | Path,
                                        output: str | Path, through_plane: bool, aha: bool) -> Path:
    """Write only missing figures into one existing common-native case bundle."""
    output, native_maps_root, prepared_subject_root, roi_bundle = (Path(value) for value in
        (output, native_maps_root, prepared_subject_root, roi_bundle))
    if not through_plane and not aha:
        raise ValueError("select at least one visual backfill: through_plane or aha")
    if not (output / "manifest.json").is_file():
        raise FileNotFoundError(f"common-native case output is missing: {output}")
    requested = {"through_plane": through_plane, "aha": aha}
    for name, enabled in requested.items():
        if enabled and (output / name).exists():
            raise FileExistsError(f"refusing existing common-native {name} output: {output / name}")

    images = load_volume_images(proposed_t1=str(proposed_t1), proposed_t2=str(proposed_t2),
                                fit_first_t1=str(fit_first_t1), fit_first_t2=str(fit_first_t2))
    native = load_cyj11_native_sax(native_maps_root / "sax", prepared_subject_root / "sax" / "observations.npz")
    staging = Path(tempfile.mkdtemp(prefix=".visual_backfill-", dir=output))
    try:
        planes = _sax_planes(native=native, images=images)
        if through_plane:
            _render_through_plane(staging=staging, subject=subject, case=case, native=native,
                                  images=images, sax_planes=planes)
        if aha:
            groups = np.asarray(native["group_idx"], int)
            shape = tuple(map(int, np.asarray(native["t1_ms"]).shape[1:]))
            myocardium = _load_scmr_masks(roi_bundle, expected_groups=groups, expected_shape=shape)
            mapping_path = staging / "aha_mapping.json"
            derive_cyj11_historical_mapping(subject=subject, case=case,
                                             prepared_sax=prepared_subject_root / "sax" / "observations.npz",
                                             roi_bundle=roi_bundle, output=mapping_path)
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            _render_aha_outputs(staging=staging, subject=subject, case=case, mapping=mapping,
                                sax_planes=planes, myocardium=myocardium, groups=groups)

        for name, enabled in requested.items():
            if enabled:
                (staging / name).rename(output / name)
        if aha:
            (staging / "aha_mapping.json").rename(output / "aha_mapping.json")
        label = "_".join(name for name, enabled in requested.items() if enabled)
        manifest = output / f"visual_backfill_{label}_manifest.json"
        write_immutable_json(manifest, {
            "schema": "common_native_visual_backfill/v1", "subject": subject, "case": case,
            "source_common_native_manifest": str((output / "manifest.json").resolve()),
            "outputs": {name: name for name, enabled in requested.items() if enabled},
            "coordinate_contract": "both final volumes are sampled on immutable acquisition-native physical planes",
            "support_equation": SUPPORT_EQUATION, "no_metrics_recomputed": True,
            "no_reconstruction": True,
        })
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "case", "native-maps-root", "prepared-subject-root", "roi-bundle",
                 "proposed-t1", "proposed-t2", "fit-first-t1", "fit-first-t2", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--through-plane", action="store_true")
    parser.add_argument("--aha", action="store_true")
    print(build_common_native_visual_backfill(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
