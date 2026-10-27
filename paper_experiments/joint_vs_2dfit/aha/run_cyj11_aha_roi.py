"""CLI for CYJ1.1 downstream ROI derivation, readiness, AHA16, and ROI reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cyj11_adapter import cyj11_native_stack_root, derive_cyj11_historical_mapping, write_cyj11_mapping
from .cyj11_aha16 import build_cyj11_aha16
from .cyj11_legacy_roi import derive_cyj11_native_roi_bundle, derive_cyj11_native_roi_config, roi_derivation_readiness
from .cyj11_myocardial_roi_metrics import build_cyj11_myocardial_roi_metrics
from .cyj11_runner import load_cyj11_endpoint_images, load_cyj11_final_pose_bundles


def _readiness(root: Path, subject: str, case: str, mapping: str | None, roi_bundle: str | None,
               sampling: str, proposed_native_bundle: str | None,
               fit_first_native_bundle: str | None) -> dict[str, str]:
    report = {"subject": subject, "case": case}
    prepared = root / "prepared" / subject / "sax" / "observations.npz"
    cache = cyj11_native_stack_root(root, subject)
    if not prepared.is_file() or not all((cache / name).is_file() for name in ("native_maps.npz", "geometry.npz", "manifest.json")):
        return {**report, "status": "INPUTS_PENDING", "reason": "prepared SAX or native SAX cache is absent"}
    try:
        if sampling == "final_pose":
            if proposed_native_bundle is None or fit_first_native_bundle is None:
                raise ValueError("both current final-pose bundle roots are required")
            load_cyj11_final_pose_bundles(proposed_native_bundle, fit_first_native_bundle, subject=subject, case=case)
        else:
            load_cyj11_endpoint_images(root, subject, case)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        return {**report, "status": "FINAL_POSE_BUNDLES_PENDING" if sampling == "final_pose" else "ENDPOINTS_PENDING", "reason": str(exc)}
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
    parser.add_argument("--mode", choices=("readiness", "derive-roi-config", "derive-roi", "derive-mapping", "write-mapping", "aha16", "myocardial"), required=True)
    parser.add_argument("--analysis-root")
    parser.add_argument("--subject")
    parser.add_argument("--case")
    parser.add_argument("--mapping")
    parser.add_argument("--roi-bundle")
    parser.add_argument("--source-roi")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument("--sampling", choices=("native_endpoint", "final_pose"), default="native_endpoint")
    parser.add_argument("--proposed-native-bundle")
    parser.add_argument("--fit-first-native-bundle")
    args = parser.parse_args()
    if args.mode == "derive-roi-config":
        if not all((args.analysis_root, args.subject, args.source_roi, args.output)):
            raise ValueError("derive-roi-config requires --analysis-root, --subject, --source-roi, and --output")
        print(derive_cyj11_native_roi_config(analysis_root=args.analysis_root, subject=args.subject,
                                              source_roi=args.source_roi, output=args.output)); return
    if args.mode == "derive-roi":
        if not args.config: raise ValueError("--config is required for derive-roi")
        if args.output is None:
            print(json.dumps(roi_derivation_readiness(args.config), indent=2)); return
        print(derive_cyj11_native_roi_bundle(config=args.config, output=args.output)); return
    if args.mode == "write-mapping":
        if not all((args.config, args.output, args.subject, args.case)):
            raise ValueError("write-mapping requires --config, --output, --subject, and --case")
        print(write_cyj11_mapping(template=args.config, output=args.output, subject=args.subject, case=args.case)); return
    if args.mode == "derive-mapping":
        if not all((args.analysis_root, args.subject, args.case, args.roi_bundle, args.output)):
            raise ValueError("derive-mapping requires --analysis-root, --subject, --case, --roi-bundle, and --output")
        root = Path(args.analysis_root)
        print(derive_cyj11_historical_mapping(subject=args.subject, case=args.case,
                                               prepared_sax=root / "prepared" / args.subject / "sax" / "observations.npz",
                                               roi_bundle=args.roi_bundle, output=args.output)); return
    if not all((args.analysis_root, args.subject, args.case)):
        raise ValueError("--analysis-root, --subject, and --case are required")
    root = Path(args.analysis_root)
    readiness = _readiness(root, args.subject, args.case, args.mapping, args.roi_bundle,
                           args.sampling, args.proposed_native_bundle, args.fit_first_native_bundle)
    if args.mode == "readiness" or args.output is None:
        print(json.dumps(readiness, indent=2)); return
    if readiness["status"] != "READY":
        raise ValueError(f"CYJ1.1 downstream execution requires READY, got {readiness['status']}: {readiness['reason']}")
    if args.mode == "aha16":
        print(build_cyj11_aha16(analysis_root=root, subject=args.subject, case=args.case, mapping_path=args.mapping,
                                roi_bundle=args.roi_bundle, output=args.output, sampling=args.sampling,
                                proposed_native_bundle=args.proposed_native_bundle,
                                fit_first_native_bundle=args.fit_first_native_bundle))
    else:
        print(build_cyj11_myocardial_roi_metrics(analysis_root=root, subject=args.subject, case=args.case, mapping_path=args.mapping,
                                                 roi_bundle=args.roi_bundle, output=args.output, sampling=args.sampling,
                                                 proposed_native_bundle=args.proposed_native_bundle,
                                                 fit_first_native_bundle=args.fit_first_native_bundle))


if __name__ == "__main__":
    main()
