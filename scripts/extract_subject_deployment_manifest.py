"""Write one explicit deployment-v1.1 manifest from a multi-subject manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def extract_subject_manifest(source: str | Path, subject: str, output: str | Path) -> Path:
    source_path, output_path = Path(source), Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_path}")
    payload: Any = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "deployment-v1.1":
        raise ValueError("source must be a deployment-v1.1 manifest")
    matches = [item for item in payload.get("subjects", []) if isinstance(item, dict) and item.get("subject_id") == subject]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one subject entry for {subject!r}, found {len(matches)}")
    result = {"schema_version": "deployment-v1.1", "subjects": matches}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(extract_subject_manifest(**vars(args)))


if __name__ == "__main__":
    main()
