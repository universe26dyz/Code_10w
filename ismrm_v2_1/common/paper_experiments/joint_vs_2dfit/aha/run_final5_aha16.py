"""CLI for explicit Final5 AHA16 readiness and immutable downstream output."""
from __future__ import annotations

import argparse
import json

from .final5_adapter import FINAL5_SUBJECTS, assess_final5_aha_cohort, assess_final5_aha_readiness
from .final5_aha16 import build_final5_aha16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-root", required=True); parser.add_argument("--subject", required=True, choices=(*FINAL5_SUBJECTS, "all")); parser.add_argument("--case", required=True)
    parser.add_argument("--roi-root", required=True); parser.add_argument("--mapping")
    parser.add_argument("--native-sax-root"); parser.add_argument("--output")
    args = parser.parse_args()
    if args.subject == "all":
        if args.output is not None or args.mapping is not None or args.native_sax_root is not None:
            raise ValueError("cohort readiness accepts no mapping, native SAX root, or output")
        print(json.dumps(assess_final5_aha_cohort(args.final_root, args.case, args.roi_root), indent=2)); return
    readiness = assess_final5_aha_readiness(args.final_root, args.subject, args.case, args.roi_root, mapping_path=args.mapping)
    if args.output is None:
        print(json.dumps(readiness, indent=2)); return
    if readiness["status"] != "READY":
        raise ValueError(f"AHA16 execution requires READY status, got {readiness['status']}: {readiness['reason']}")
    if args.native_sax_root is None:
        raise ValueError("--native-sax-root is required when --output is provided")
    print(build_final5_aha16(final_root=args.final_root, subject=args.subject, case=args.case, roi_root=args.roi_root,
                             mapping_path=args.mapping, native_sax_root=args.native_sax_root, output=args.output))


if __name__ == "__main__":
    main()
