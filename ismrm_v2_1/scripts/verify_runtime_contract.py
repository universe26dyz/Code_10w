#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--decoder", type=Path)
    p.add_argument("--b6-model", type=Path)
    p.add_argument("--protocol", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "runtime/runtime_contract.json").read_text())
    paths = {
        "decoder_sha256": args.decoder or root / "runtime/model_assets/decoder_scmr_v5_10_100.pth",
        "b6_model_sha256": args.b6_model or root / "runtime/model_assets/b6_model.pt",
        "protocol_sha256": args.protocol or root / "configs/protocol_scmr_v5_10_100.yaml",
    }
    for key, path in paths.items():
        path = Path(path)
        expected = contract[key]
        if not path.is_file():
            raise SystemExit(f"missing runtime asset: {path}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{key} mismatch: expected {expected}, got {actual}")
    if contract["protocol"]["ti_ms"] != [10.0, 100.0]:
        raise SystemExit("protocol contract is not SCMR-v5 10/100")
    print("RUNTIME CONTRACT: PASS")

if __name__ == "__main__":
    main()

