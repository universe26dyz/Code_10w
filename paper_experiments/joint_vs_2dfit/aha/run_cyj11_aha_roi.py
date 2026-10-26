"""CLI for CYJ1.1 downstream ROI derivation, readiness, AHA16, and ROI reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cyj11_adapter import write_cyj11_mapping
from .cyj11_aha16 import build_cyj11_aha16
from .cyj11_legacy_roi import derive_cyj11_native_roi_bundle, roi_derivation_readiness
from .cyj11_myocardial_roi_metrics import build_cyj11_myocardial_roi_metrics
from .cyj11_runner import load_cyj11_endpoint_images


def _readiness(root: Path, subject: str, case: str, mapping: str | None, roi_bundle: str | None) -> dict[str, str]:
    report = {"subject": subject, "case": case}
    prepared = root / "prepared" / subject / "sax" / "observations.npz"
    cache = root / "native_cache" / subject / "sax"
    if not prepared.is_file() or not all((cache / name).is_file() for name in ("native_maps.npz", "geometry.npz", "manifest.json")):
        return {**report, "status": "INPUTS_PENDING", "reason": "prepared SAX or native SAX cache is absent"}
    try:
        load_cyj11_endpoint_images(root, subject, case)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        return {**report, "status": "ENDPOINTS_PENDING", "reason": str(exc)}
    if mapping is None or roi_bundle is None:
        return {**report, "status": "ROI_MAPPING_PENDING", "reason": "explicit mapping and immutable ROI bundle are required"}
    from .cyj11_adapter import require_cyj11_mapping
    try:
        require_cyj11_mapping(mapping, subject=subject, case=case, prepared_sax_path=prepared,
                              roi_bundle_path=Path(roi_bundle) / "roi_derivation_manifest.json")
    except (FileNotFoundError, ValueError) as exc:
        return {**report, "status": "ROI_MAPPING_INVALID", "reason": str(exc)}
    return {**report, "status": "READY"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("readiness", "derive-roi", "write-mapping", "aha16", "myocardial"), required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--subject")
    parser.add_argument("--case")
    parser.add_argument("--mapping")
    parser.add_argument("--roi-bundle")
    parser.add_argument("--config")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "derive-roi":
        if not args.config: raise ValueError("--config is required for derive-roi")
        if args.output is None:
            print(json.dumps(roi_derivation_readiness(args.config), indent=2)); return
        print(derive_cyj11_native_roi_bundle(config=args.config, output=args.output)); return
    if args.mode == "write-mapping":
        if not all((args.config, args.output, args.subject, args.case)):
            raise ValueError("write-mapping requires --config, --output, --subject, and --case")
        print(write_cyj11_mapping(template=args.config, output=args.output, subject=args.subject, case=args.case)); return
    if not all((args.analysis_root, args.subject, args.case)):
        raise ValueError("--analysis-root, --subject, and --case are required")
    root = Path(args.analysis_root)
    readiness = _readiness(root, args.subject, args.case, args.mapping, args.roi_bundle)
    if args.mode == "readiness" or args.output is None:
        print(json.dumps(readiness, indent=2)); return
    if readiness["status"] != "READY":
        raise ValueError(f"CYJ1.1 downstream execution requires READY, got {readiness['status']}: {readiness['reason']}")
    if args.mode == "aha16":
        print(build_cyj11_aha16(analysis_root=root, subject=args.subject, case=args.case, mapping_path=args.mapping,
                                roi_bundle=args.roi_bundle, output=args.output))
    else:
        print(build_cyj11_myocardial_roi_metrics(analysis_root=root, subject=args.subject, case=args.case, mapping_path=args.mapping,
                                                 roi_bundle=args.roi_bundle, output=args.output))


if __name__ == "__main__":
    main()
