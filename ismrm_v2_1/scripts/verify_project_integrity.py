#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

MODULES = [
    "01_RAW_DICOM", "02_PREPROCESS_MATLAB", "03_PREPARE_OBSERVATIONS",
    "04_SPARSE_PROFILES", "05_SCMR_V5_FIT_FIRST", "06_PROPOSED",
    "07_FINAL_POSE", "08_EVALUATION_AND_VISUALIZATION",
    "09_INTEGRITY_AND_REPORTING",
]

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path, default=Path(__file__).resolve().parents[1], nargs="?")
    args = p.parse_args()
    root = args.root.resolve()
    for module in MODULES:
        path = root / "pipeline" / module
        for name in ("README_zh-CN.md", "README_en.md"):
            if not (path / name).is_file():
                raise SystemExit(f"missing module manual: {path / name}")
    for directory in ("configs", "common", "runtime", "subjects", "runs", "docs", "tests"):
        if not (root / directory).is_dir():
            raise SystemExit(f"missing project directory: {root / directory}")
    json.loads((root / "runtime/runtime_contract.json").read_text())
    print(f"PROJECT INTEGRITY: PASS ({len(MODULES)} modules)")

if __name__ == "__main__":
    main()

