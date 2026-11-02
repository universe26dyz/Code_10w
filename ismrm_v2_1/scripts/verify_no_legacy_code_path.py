#!/usr/bin/env python3
from pathlib import Path
import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    forbidden = "/data/" + "dengyz/code/Code_10w"
    suffixes = {".py", ".sh", ".m", ".md", ".json", ".yaml", ".yml"}
    hits = []
    for path in args.root.rglob("*"):
        if path.is_file() and path.name != "verify_no_legacy_code_path.py" and path.suffix in suffixes:
            if forbidden in path.read_text(errors="ignore"):
                hits.append(str(path))
    if hits:
        raise SystemExit("legacy absolute path found:\n" + "\n".join(hits))
    print("NO LEGACY CODE_10W PATH: PASS")

if __name__ == "__main__":
    main()
