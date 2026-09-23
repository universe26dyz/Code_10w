"""Create a formal-MPL timing manifest only from exact prepared manifest entries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.deployment_manifest import load_deployment_manifest


def build_formal_manifest(deployment_manifest: str | Path, prepared_root: str | Path, output_path: str | Path) -> dict[str, object]:
    output_path, prepared_root = Path(output_path), Path(prepared_root)
    if output_path.exists():
        raise FileExistsError(f"Formal manifest output already exists: {output_path}")
    if not prepared_root.is_dir():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared_root}")
    sources = []
    for entry in load_deployment_manifest(deployment_manifest):
        observations = prepared_root / entry["subject_id"] / entry["stack"] / "observations.npz"
        if not observations.is_file():
            raise FileNotFoundError(f"Required prepared observations are missing: {observations}")
        sources.append({"subject_id": entry["subject_id"], "stack": entry["stack"], "observations": str(observations)})
    payload = {"schema_version": "formal-timing-manifest-v1", "sources": sources}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_count": len(sources), "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(build_formal_manifest(args.deployment_manifest, args.prepared_root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
