"""Create one case-specific historical AHA mapping from a verified native ROI bundle."""
from __future__ import annotations

import argparse

from .cyj11_adapter import derive_cyj11_historical_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--prepared-sax", required=True)
    parser.add_argument("--roi-bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(derive_cyj11_historical_mapping(
        subject=args.subject, case=args.case, prepared_sax=args.prepared_sax,
        roi_bundle=args.roi_bundle, output=args.output,
    ))


if __name__ == "__main__":
    main()
