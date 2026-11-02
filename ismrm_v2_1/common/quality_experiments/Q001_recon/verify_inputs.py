"""CLI for local or server-side read-only Q001 input validation."""

from __future__ import annotations

import argparse
import json

from .contracts import verify_q001_input_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    print(json.dumps(verify_q001_input_bundle(parser.parse_args().input_root), indent=2))


if __name__ == "__main__": main()
