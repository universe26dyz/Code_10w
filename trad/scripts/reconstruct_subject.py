"""Run one explicit three-stack Trad reconstruction from prepared observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_training import run_reconstruction


def reconstruct_subject(prepared_root: str | Path, subject_id: str, config: str | Path, output: str | Path) -> dict[str, object]:
    root = Path(prepared_root)
    observations = [root / subject_id / stack / "observations.npz" for stack in ("sax", "2ch", "4ch")]
    missing = [str(path) for path in observations if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required exact prepared stack observations are missing: " + "; ".join(missing))
    method_root = Path(__file__).resolve().parents[1]
    return run_reconstruction(config, method_root / "configs" / "protocol_hhz_v1.yaml", observations, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True); parser.add_argument("--subject-id", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(reconstruct_subject(args.prepared_root, args.subject_id, args.config, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
