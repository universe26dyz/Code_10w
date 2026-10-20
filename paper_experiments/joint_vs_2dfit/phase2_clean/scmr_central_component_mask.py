"""Build read-only SCMR-style support masks on already co-registered planes.

The historical SCMR support is reconstructed from its six per-plane inputs,
not inferred from a rendered bitmap.  ``phase4r_l100_central_component`` is a
control that applies only the legacy component rule to the Phase4R L100 mask.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .contracts import sha256, write_immutable_json


STRICT_COMMON_INPUTS = ("original", "registered", "synthetic", "training_mask", "synthetic_support", "fitting_valid_mask")
REQUIRED_INPUTS = STRICT_COMMON_INPUTS + ("phase4r_l100",)
FORMAL_MASKS = ("scmr_style_central_component", "phase4r_l100", "phase4r_l100_central_component")
_PLANE_PATH = re.compile(r"^planes/(t1|t2)/(sax|2ch|4ch)/group([0-9]+)\.npz$")


def _same_shape(**arrays: np.ndarray) -> tuple[int, ...]:
    shapes = {np.asarray(value).shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("all SCMR support inputs must share one shape")
    return next(iter(shapes))


def build_legacy_strict_common(original: np.ndarray, registered: np.ndarray, synthetic: np.ndarray, training_mask: np.ndarray, synthetic_support: np.ndarray, fitting_valid_mask: np.ndarray) -> np.ndarray:
    """Return the historical pre-component support for one native plane."""
    _same_shape(original=original, registered=registered, synthetic=synthetic, training_mask=training_mask, synthetic_support=synthetic_support, fitting_valid_mask=fitting_valid_mask)
    return (np.asarray(training_mask, bool) & np.asarray(synthetic_support, bool) & np.asarray(fitting_valid_mask, bool)
            & np.isfinite(np.asarray(original, float)) & np.isfinite(np.asarray(registered, float)) & np.isfinite(np.asarray(synthetic, float)))


def select_scmr_central_component(support: np.ndarray, *, minimum_ratio: float = 0.25) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the historical 8-connected per-plane central-component rule."""
    support = np.asarray(support, bool)
    if support.ndim != 2:
        raise ValueError("SCMR central-component selection requires one 2-D plane")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    labels, count = ndimage.label(support, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(support), {"component_count": 0, "selected_label": None, "selection": "empty_support", "minimum_ratio": float(minimum_ratio)}
    areas = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(areas) + 1)
    center_rc = tuple(int(round((size - 1) / 2.0)) for size in support.shape)
    center_label = int(labels[center_rc])
    if center_label and areas[center_label - 1] >= minimum_ratio * areas[largest_label - 1]:
        selected_label, selection = center_label, "contains_image_center"
    else:
        selected_label, selection = largest_label, "largest_component"
    return labels == selected_label, {"component_count": int(count), "selected_label": int(selected_label), "selection": selection, "center_rc": list(center_rc), "selected_area": int(areas[selected_label - 1]), "largest_area": int(areas[largest_label - 1]), "minimum_ratio": float(minimum_ratio), "connectivity": 8}


def build_sensitivity_masks(payload: dict[str, np.ndarray], *, minimum_ratio: float = 0.25) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build SCMR and the Phase4R selection-rule control from one bundle."""
    missing = [name for name in REQUIRED_INPUTS if name not in payload]
    if missing:
        raise ValueError(f"input bundle is missing required arrays: {', '.join(missing)}")
    strict = build_legacy_strict_common(**{name: payload[name] for name in STRICT_COMMON_INPUTS})
    scmr, scmr_record = select_scmr_central_component(strict, minimum_ratio=minimum_ratio)
    l100 = np.asarray(payload["phase4r_l100"], bool)
    _same_shape(strict=strict, phase4r_l100=l100)
    l100_central, l100_record = select_scmr_central_component(l100, minimum_ratio=minimum_ratio)
    return {"scmr_style_central_component": scmr, "phase4r_l100": l100, "phase4r_l100_central_component": l100_central}, {"legacy_strict_common": "training_mask AND synthetic_support AND fitting_valid_mask AND finite(original) AND finite(registered) AND finite(synthetic)", "scmr_style_central_component": scmr_record, "phase4r_l100_central_component": l100_record}


def _plane_identity(relative: Path) -> tuple[str, str, int]:
    match = _PLANE_PATH.fullmatch(relative.as_posix())
    if match is None:
        raise ValueError(f"formal SCMR masks require planes/{{t1,t2}}/{{sax,2ch,4ch}}/groupNNN.npz, got {relative}")
    return match.group(1), match.group(2), int(match.group(3))


def load_validated_scmr_mask_plane(mask_root: str | Path, *, parameter: str, stack: str, original_group_idx: int, shape_rc: tuple[int, int], source_support: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a formal SCMR artifact only after identity, hash and source checks."""
    root = Path(mask_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "scmr_central_component_mask/v2":
        raise ValueError("formal native support requires scmr_central_component_mask/v2")
    records = [record for record in manifest.get("planes", []) if record.get("parameter") == parameter and record.get("stack") == stack and int(record.get("original_group_idx", -1)) == int(original_group_idx)]
    if len(records) != 1:
        raise ValueError(f"SCMR mask manifest lacks exactly one {parameter}/{stack}/group{original_group_idx:03d}")
    record = records[0]
    target = root / record["output"]["path"]
    expected_relative = Path("planes") / parameter / stack / f"group{original_group_idx:03d}.npz"
    if target.relative_to(root) != expected_relative or not target.is_file() or sha256(target) != record["output"].get("sha256"):
        raise ValueError(f"SCMR mask artifact provenance mismatch for {parameter}/{stack}/group{original_group_idx:03d}")
    source = Path(record["input"]["path"])
    if not source.is_file() or sha256(source) != record["input"].get("sha256"):
        raise ValueError(f"SCMR source provenance unavailable for {parameter}/{stack}/group{original_group_idx:03d}")
    if tuple(record.get("shape_rc", ())) != tuple(shape_rc):
        raise ValueError(f"SCMR mask shape provenance mismatch for {parameter}/{stack}/group{original_group_idx:03d}")
    with np.load(target, allow_pickle=False) as payload:
        missing = [name for name in (*FORMAL_MASKS, "source_support") if name not in payload.files]
        if missing:
            raise ValueError(f"{target}: missing formal SCMR arrays {missing}")
        values = {name: np.asarray(payload[name], bool) for name in FORMAL_MASKS}
        artifact_source = np.asarray(payload["source_support"], bool)
    if any(value.shape != tuple(shape_rc) for value in (*values.values(), artifact_source)):
        raise ValueError(f"{target}: SCMR mask shape does not match native reference")
    if not np.array_equal(artifact_source, np.asarray(source_support, bool)):
        raise ValueError(f"{target}: SCMR source_support does not match the selected R0 plane")
    return values, {"mask_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}, "mask_plane": {"path": str(target.resolve()), "sha256": sha256(target)}, "source_input": record["input"], "primary_mask": "scmr_style_central_component"}


def run_scmr_central_component_mask(input_root: str | Path, output: str | Path, *, minimum_ratio: float = 0.25) -> Path:
    """Publish sensitivity-ready bundles without touching formal outputs."""
    input_root, output = Path(input_root), Path(output)
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    files = sorted(input_root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz input bundles under {input_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records = []
        for source in files:
            relative = source.relative_to(input_root)
            parameter, stack, group = _plane_identity(relative)
            with np.load(source, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]) for name in archive.files}
            if "source_support" not in payload:
                raise ValueError(f"{source}: formal SCMR artifact requires source_support provenance")
            masks, record = build_sensitivity_masks(payload, minimum_ratio=minimum_ratio)
            target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
            preserved = {name: payload[name] for name in ("reference", "prediction", "source_support") if name in payload}
            np.savez_compressed(target, **preserved, **masks)
            records.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "shape_rc": list(np.asarray(payload["source_support"]).shape), "input": {"path": str(source.resolve()), "sha256": sha256(source)}, "output": {"path": str(target.relative_to(staging)), "sha256": sha256(target)}, "component_audit": record})
        write_immutable_json(staging / "manifest.json", {"schema": "scmr_central_component_mask/v2", "planes": records, "required_arrays": list(REQUIRED_INPUTS), "preserved_arrays": ["reference", "prediction", "source_support"], "minimum_ratio": float(minimum_ratio), "primary_mask": "scmr_style_central_component", "sensitivity_masks": ["phase4r_l100", "phase4r_l100_central_component"], "no_training": True, "no_reconstruction": True, "no_export": True, "no_reprojection": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True); parser.add_argument("--output", required=True); parser.add_argument("--minimum-ratio", type=float, default=0.25)
    print(run_scmr_central_component_mask(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
