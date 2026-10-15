"""Data-dependent Phase-1.2 audits, executed after real XCAT input is available."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def quantitative_tuple_audit(t1, t2, pd, lv_mask, phantom_spec: str | Path, tolerance: float = 1e-5) -> dict:
    spec = json.loads(Path(phantom_spec).read_text())
    declared = {tuple(float(item[key]) for key in ("t1_ms", "t2_ms", "pd")): name for name, item in spec["structures"].items()}
    values = np.stack((np.asarray(t1), np.asarray(t2), np.asarray(pd)), -1).reshape(-1, 3)
    unique, counts = np.unique(values, axis=0, return_counts=True)
    entries, unmatched = [], 0
    for value, count in zip(unique, counts):
        matches = [name for tup, name in declared.items() if np.allclose(value, tup, atol=tolerance, rtol=0)]
        background = np.allclose(value, 0, atol=tolerance, rtol=0)
        unmatched += 0 if matches or background else int(count)
        entries.append({"tuple": value.tolist(), "voxel_count": int(count), "matching_declared_structures": matches, "background": bool(background)})
    lv_tuple = (1410.0, 50.0, float(spec["structures"]["lv_myocardium"]["pd"]))
    lv_ok = bool(np.allclose(np.stack((t1[lv_mask], t2[lv_mask], pd[lv_mask]), -1), lv_tuple, atol=tolerance, rtol=0))
    if unmatched or not lv_ok: raise ValueError("quantitative tuple audit failed")
    return {"quantitative_tuple_gate": True, "unmatched_voxels": unmatched, "tuples": entries, "lv_mask_frozen_tuple_gate": bool(lv_ok)}

def structure_id_audit_status() -> dict:
    return {"structure_id_audit_gate": "UNVERIFIED", "reason": "requires licensed XCAT version-specific audit run"}
