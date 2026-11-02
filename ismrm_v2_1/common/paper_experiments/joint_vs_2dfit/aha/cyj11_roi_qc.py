"""Render immutable native-SAX manual-ROI quality-control overlays.

This is a visualization-only downstream tool.  It reads an existing native
SAX cache and immutable ROI bundle, then writes a fresh directory of PNG/PDF
figures without changing either input.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_adapter import cyj11_native_stack_root, load_cyj11_native_sax


SCHEMA = "cyj11_native_sax_roi_qc/v1"
REQUIRED_MASKS = ("outer_filled", "inner_filled", "myocardium_full", "landmark_labels")


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _load_masks(bundle: Path, groups: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    manifest_path, archive_path = bundle / "manifest.json", bundle / "sax_myocardium_masks.npz"
    if not manifest_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("ROI QC requires immutable manifest.json and sax_myocardium_masks.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exact_native_myocardium_bundle/v2" or manifest.get("status") != "PASS":
        raise ValueError("ROI QC requires a PASS exact_native_myocardium_bundle/v2 bundle")
    with np.load(archive_path, allow_pickle=False) as archive:
        if "group_idx" not in archive or any(name not in archive for name in REQUIRED_MASKS):
            raise ValueError("ROI QC mask archive is incomplete")
        saved_groups = np.asarray(archive["group_idx"], int)
        masks = {name: np.asarray(archive[name]) for name in REQUIRED_MASKS}
    if not np.array_equal(saved_groups, groups):
        raise ValueError("ROI QC bundle group identity mismatches native SAX")
    expected = (len(groups), *shape)
    if any(mask.shape != expected for mask in masks.values()):
        raise ValueError("ROI QC bundle shape mismatches native SAX")
    return masks


def _render_plane(output: Path, *, image: np.ndarray, outer: np.ndarray, inner: np.ndarray,
                  myocardium: np.ndarray, landmarks: np.ndarray, subject: str, group: int,
                  parameter: str) -> None:
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    if output.exists() or pdf.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {pdf}")
    image = np.asarray(image, float)
    finite = image[np.isfinite(image)]
    if not finite.size:
        raise ValueError(f"ROI QC cannot render all-nonfinite {parameter} image for group {group}")
    vmin, vmax = np.percentile(finite, (1, 99))
    if not vmin < vmax:
        vmin, vmax = float(finite.min()) - .5, float(finite.max()) + .5
    figure, axis = plt.subplots(figsize=(6, 5.5), constrained_layout=True)
    rendered = axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    overlay = np.ma.masked_where(~np.asarray(myocardium, bool), np.ones_like(image))
    axis.imshow(overlay, cmap=ListedColormap(["gold"]), alpha=.22, origin="upper")
    for mask, color, label in ((outer, "lime", "outer"), (inner, "cyan", "inner")):
        mask = np.asarray(mask, bool)
        if mask.any() and not mask.all():
            axis.contour(mask.astype(float), levels=(.5,), colors=color, linewidths=1.2)
            axis.plot([], [], color=color, label=label)
    landmarks = np.asarray(landmarks, int)
    for label in sorted(int(value) for value in np.unique(landmarks) if value > 0):
        rows, columns = np.where(landmarks == label)
        axis.scatter(columns, rows, s=22, c="red", edgecolors="white", linewidths=.4, zorder=4)
        axis.text(float(columns.mean()) + .8, float(rows.mean()) + .8, str(label), color="red", fontsize=8,
                  weight="bold", zorder=5)
    axis.set_title(f"{subject} native SAX group {group}: {parameter.upper()} with manual ROI")
    axis.set_xlabel("column"); axis.set_ylabel("row")
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(loc="lower right", fontsize=8, framealpha=.75)
    figure.colorbar(rendered, ax=axis, label=f"{parameter.upper()} (ms)", shrink=.84)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_native_roi_qc(*, native: Mapping[str, np.ndarray], masks: Mapping[str, np.ndarray], subject: str,
                        output: str | Path, manifest_updates: Mapping[str, Any] | None = None) -> Path:
    """Write T1/T2 manual-ROI overlays for every native SAX group."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    groups = np.asarray(native["group_idx"], int)
    t1, t2 = np.asarray(native["t1_ms"], float), np.asarray(native["t2_ms"], float)
    if t1.ndim != 3 or t2.shape != t1.shape or t1.shape[0] != len(groups):
        raise ValueError("ROI QC native T1/T2 maps must be equal-shape SAX stacks")
    expected = t1.shape
    if any(np.asarray(masks[name]).shape != expected for name in REQUIRED_MASKS):
        raise ValueError("ROI QC masks must match the native SAX stack shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for index, group in enumerate(groups):
            for parameter, image in (("t1", t1[index]), ("t2", t2[index])):
                _render_plane(staging / subject / f"group{int(group):03d}_{parameter}_roi_qc.png", image=image,
                              outer=masks["outer_filled"][index], inner=masks["inner_filled"][index],
                              myocardium=masks["myocardium_full"][index], landmarks=masks["landmark_labels"][index],
                              subject=subject, group=int(group), parameter=parameter)
        manifest: dict[str, Any] = {
            "schema": SCHEMA, "subject": subject, "groups": [int(group) for group in groups],
            "figures_per_group": ["t1_roi_qc.png/pdf", "t2_roi_qc.png/pdf"],
            "overlay": {"background": "native fitted map", "gold": "myocardium_full",
                        "lime": "manual outer contour", "cyan": "manual inner contour", "red": "manual landmark label"},
            "downstream_only": True, "no_reconstruction": True, "no_formal_metric": True, "immutable_output": True,
        }
        if manifest_updates:
            manifest.update(dict(manifest_updates))
        write_immutable_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def build_native_roi_qc(*, analysis_root: str | Path, subject: str, roi_bundle: str | Path, output: str | Path) -> Path:
    """Load verified native SAX and ROI inputs, then write a fresh QC directory."""
    root, bundle = Path(analysis_root), Path(roi_bundle)
    prepared = root / "prepared" / subject / "sax" / "observations.npz"
    native_root = cyj11_native_stack_root(root, subject)
    native = load_cyj11_native_sax(native_root, prepared)
    masks = _load_masks(bundle, np.asarray(native["group_idx"], int), tuple(map(int, native["t1_ms"].shape[1:])))
    return write_native_roi_qc(native=native, masks=masks, subject=subject, output=output, manifest_updates={
        "prepared_sax": _record(prepared),
        "native_sax": {name: _record(native_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
        "roi_bundle": {"manifest": _record(bundle / "manifest.json"),
                       "archive": _record(bundle / "sax_myocardium_masks.npz")},
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--roi-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_native_roi_qc(analysis_root=args.analysis_root, subject=args.subject,
                              roi_bundle=args.roi_bundle, output=args.output))


if __name__ == "__main__":
    main()
