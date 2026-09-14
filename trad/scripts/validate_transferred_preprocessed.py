"""Validate transferred MATLAB-v7.3 outputs; this never invokes MATLAB preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.deployment_manifest import load_deployment_manifest
from modules.module_02_data_bridge.mat_v73 import load_preprocessed_v73


def _mat_path(root: Path, subject_id: str, stack: str) -> Path:
    return root / subject_id / stack / "preprocessed.mat"


def validate_transferred_preprocessed(manifest_path: str | Path, preprocessed_root: str | Path, output_path: str | Path) -> dict[str, object]:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Transfer QC output already exists: {output_path}")
    root = Path(preprocessed_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Preprocessed root does not exist: {root}")
    entries = load_deployment_manifest(manifest_path)
    results = []
    for entry in entries:
        path = _mat_path(root, entry["subject_id"], entry["stack"])
        data = load_preprocessed_v73(path)
        results.append({"subject_id": entry["subject_id"], "stack": entry["stack"], "preprocessed_mat": str(path), "mag_crop_shape": list(data["mag_crop"].shape), "tr_ms": data["tr_ms"], "vps": data["vps"]})
    report: dict[str, object] = {"schema_version": "transfer-qc-v1", "entry_count": len(results), "entries": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_transferred_preprocessed(args.manifest, args.preprocessed_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
