"""Create or inspect one explicit Final5 cropped ROI bundle from legacy ROI."""
from __future__ import annotations

import argparse
import json

from .final5_legacy_roi import derivation_readiness, derive_final5_cropped_roi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.output is None:
        print(json.dumps(derivation_readiness(args.config), indent=2)); return
    print(derive_final5_cropped_roi(config=args.config, output=args.output))


if __name__ == "__main__":
    main()
