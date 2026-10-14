"""Immutable artifact and hash closure for a completed Phase-1.2 dataset."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def finalize_phase1_2(output_root: str | Path) -> Path:
    root = Path(output_root); required = ("gt_maps.npz", "geometry_qc.json", "geometry_alignment_qc.json", "clean_signal_field.npz", "timing_selection.json", "sax/observations.npz", "2ch/observations.npz", "4ch/observations.npz", "synthetic_source_manifest.json")
    for item in required:
        if not (root / item).is_file(): raise FileNotFoundError(f"required Phase-1.2 artifact is missing: {item}")
    source, qc, alignment = (json.loads((root / name).read_text()) for name in ("synthetic_source_manifest.json", "geometry_qc.json", "geometry_alignment_qc.json"))
    if source.get("scientific_gt_geometry_gate") is not True or qc.get("scientific_gt_geometry_gate") is not True: raise ValueError("Phase-1.2 GT geometry gate is not passing")
    if source.get("external_reference_provenance", {}).get("reference_commit") != "97558db05f4e344865f472de974b9f4d042a1a15": raise ValueError("Phase-1.2 source is not anchored to the required Hanson commit")
    if alignment.get("lv_alignment_gate") is not True: raise ValueError("Phase-1.2 LV alignment gate is not passing")
    maps, signal = np.load(root / "gt_maps.npz"), np.load(root / "clean_signal_field.npz"); myo, support = maps["myocardium_mask"], maps["signal_support_mask"]
    if not np.array_equal(maps["spacing_mm"], np.ones(3)) or str(maps["axis_order"]) != "xyz" or str(maps["world_convention"]) != "LPS": raise ValueError("formal GT is not canonical xyz/LPS/1-mm")
    if not (np.all(maps["t1_ms"][myo] == 1410.0) and np.all(maps["t2_ms"][myo] == 50.0) and np.all(maps["b1"][myo] == 1.0)): raise ValueError("frozen LV quantitative values failed")
    if not np.array_equal(signal["signals"][~support], np.zeros((int((~support).sum()), 10), dtype=np.float32)): raise ValueError("signal background is not exactly zero")
    if source.get("spatial_integration", {}).get("inplane_samples_row_col") != [6, 6] or source["spatial_integration"].get("thickness_samples") != 9: raise ValueError("formal renderer is not pinned to 6x6x9")
    for stack in ("sax", "2ch", "4ch"):
        obs = np.load(root / stack / "observations.npz")
        if obs["images"].shape[0] % 10 or not np.all(obs["tr_ms"] == 2.61) or not np.all(obs["vps"] == 87) or not np.all(np.isfinite(obs["images"])): raise ValueError(f"invalid {stack} observation contract")
    final = root / "phase1_2_final_manifest.json"
    if final.exists(): raise FileExistsError(f"refusing to overwrite {final}")
    final.write_text(json.dumps({"phase": "1.2", "formal_synthetic_ready": True, "artifact_sha256": {item: _sha256(root / item) for item in required[:-1]}, "raw_xcat_volume_sha256": {name: record["sha256"] for name, record in source["xcat"]["raw_volumes"].items()}}, indent=2, sort_keys=True) + "\n")
    return final

def assert_final_manifest_ready(root: str | Path) -> dict:
    root = Path(root); manifest = json.loads((root / "phase1_2_final_manifest.json").read_text())
    if manifest.get("formal_synthetic_ready") is not True: raise ValueError("Phase-1.2 formal_synthetic_ready is not true")
    for item, digest in manifest.get("artifact_sha256", {}).items():
        if _sha256(root / item) != digest: raise ValueError(f"Phase-1.2 final artifact SHA mismatch: {item}")
    return manifest
