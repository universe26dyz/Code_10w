"""Build read-only SCMR-style support masks on already co-registered planes.

The historical SCMR support is reconstructed from its six per-plane inputs,
not inferred from a rendered bitmap.  ``phase4r_l100_central_component`` is a
control that applies only the legacy component rule to the Phase4R L100 mask.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .contracts import sha256, write_immutable_json


STRICT_COMMON_INPUTS = ("original", "registered", "synthetic", "training_mask", "synthetic_support", "fitting_valid_mask")
REQUIRED_INPUTS = STRICT_COMMON_INPUTS + ("phase4r_l100",)


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
            with np.load(source, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]) for name in archive.files}
            masks, record = build_sensitivity_masks(payload, minimum_ratio=minimum_ratio)
            target = staging / source.relative_to(input_root); target.parent.mkdir(parents=True, exist_ok=True)
            preserved = {name: payload[name] for name in ("reference", "prediction", "source_support") if name in payload}
            np.savez_compressed(target, **preserved, **masks)
            records.append({"input": {"path": str(source.resolve()), "sha256": sha256(source)}, "output": str(target.relative_to(staging)), "component_audit": record})
        write_immutable_json(staging / "manifest.json", {"schema": "scmr_central_component_mask/v1", "inputs": records, "required_arrays": list(REQUIRED_INPUTS), "optional_preserved_arrays": ["reference", "prediction", "source_support"], "minimum_ratio": float(minimum_ratio), "no_training": True, "no_reconstruction": True, "no_export": True, "no_reprojection": True})
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
