"""CLI for immutable local Final5 native-SAX myocardial ROI statistics."""
from __future__ import annotations

import argparse
import json

from .final5_adapter import FINAL5_SUBJECTS
from .final5_myocardial_roi_metrics import (assess_final5_myocardial_roi_readiness,
                                             build_final5_myocardial_roi_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--subject", required=True, choices=FINAL5_SUBJECTS)
    parser.add_argument("--case", required=True)
    parser.add_argument("--roi-root", required=True)
    parser.add_argument("--mapping")
    parser.add_argument("--standardization-manifest")
    parser.add_argument("--native-sax-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    readiness = assess_final5_myocardial_roi_readiness(
        args.final_root, args.subject, args.case, args.roi_root, mapping_path=args.mapping,
        standardization_manifest_path=args.standardization_manifest)
    if args.output is None:
        print(json.dumps(readiness, indent=2)); return
    if readiness["status"] != "READY":
        raise ValueError(f"myocardial ROI execution requires READY status, got {readiness['status']}: {readiness['reason']}")
    if args.mapping is None or args.native_sax_root is None:
        raise ValueError("--mapping and --native-sax-root are required when --output is provided")
    print(build_final5_myocardial_roi_metrics(
        final_root=args.final_root, subject=args.subject, case=args.case,
        roi_root=args.roi_root, mapping_path=args.mapping,
        native_sax_root=args.native_sax_root, output=args.output,
        standardization_manifest_path=args.standardization_manifest))


if __name__ == "__main__":
    main()
