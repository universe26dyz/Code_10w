"""Run paired final-pose AHA16 and myocardial ROI evaluation from explicit paths.

The command is intentionally independent of the historical unified-root layout.
It is suitable for one completed HHZ sparse case and never starts a
reconstruction, mapping fit, or model query.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_aha16 import build_cyj11_aha16
from .cyj11_myocardial_roi_metrics import build_cyj11_myocardial_roi_metrics
from .paired_final_pose_lv_metrics import build_paired_final_pose_lv_metrics


SCHEMA = "paired_final_pose_aha_myocardial_postprocess/v1"
SUPPORT_EQUATION = "manual anatomical ROI AND finite(reference) AND proposed-valid AND fit-first-valid AND finite(proposed, fit-first)"


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def build_paired_final_pose_postprocess(*, subject: str, case: str, native_sax_root: str | Path,
                                        prepared_sax: str | Path, mapping: str | Path, roi_bundle: str | Path,
                                        proposed_bundle: str | Path, fit_first_bundle: str | Path,
                                        output: str | Path,
                                        outer_mask_bundle: str | Path | None = None) -> Path:
    """Create one immutable paired AHA16/ROI bundle for a completed sparse case."""
    native_sax_root, prepared_sax, mapping, roi_bundle = (Path(value) for value in (native_sax_root, prepared_sax, mapping, roi_bundle))
    proposed_bundle, fit_first_bundle, output = (Path(value) for value in (proposed_bundle, fit_first_bundle, output))
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    for path in (native_sax_root / "native_maps.npz", native_sax_root / "geometry.npz", native_sax_root / "manifest.json",
                 prepared_sax, mapping, roi_bundle / "roi_derivation_manifest.json",
                 proposed_bundle / "manifest.json", fit_first_bundle / "manifest.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        # analysis_root is unused for final_pose after explicit roots are passed;
        # retain a stable value for backward-compatible writer signatures.
        placeholder_root = staging / "explicit_inputs"
        aha = build_cyj11_aha16(
            analysis_root=placeholder_root, subject=subject, case=case, mapping_path=mapping,
            roi_bundle=roi_bundle, output=staging / "aha16", sampling="final_pose",
            proposed_native_bundle=proposed_bundle, fit_first_native_bundle=fit_first_bundle,
            native_sax_root=native_sax_root, prepared_sax=prepared_sax,
        )
        myocardial = build_cyj11_myocardial_roi_metrics(
            analysis_root=placeholder_root, subject=subject, case=case, mapping_path=mapping,
            roi_bundle=roi_bundle, output=staging / "myocardial_roi", sampling="final_pose",
            proposed_native_bundle=proposed_bundle, fit_first_native_bundle=fit_first_bundle,
            native_sax_root=native_sax_root, prepared_sax=prepared_sax,
        )
        lv_outer = None
        if outer_mask_bundle is not None:
            lv_outer = build_paired_final_pose_lv_metrics(
                subject=subject, case=case, native_sax_root=native_sax_root, prepared_sax=prepared_sax,
                outer_mask_bundle=outer_mask_bundle, proposed_bundle=proposed_bundle,
                fit_first_bundle=fit_first_bundle, output=staging / "lv_outer_filled",
            )
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "sampling": "final_pose",
            "support_equation": SUPPORT_EQUATION,
            "native_sax": {name: _record(native_sax_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "prepared_sax": _record(prepared_sax), "mapping": _record(mapping),
            "roi_bundle": _record(roi_bundle / "roi_derivation_manifest.json"),
            "proposed_final_pose_bundle": _record(proposed_bundle / "manifest.json"),
            "fit_first_final_pose_bundle": _record(fit_first_bundle / "manifest.json"),
            "outputs": {"aha16_manifest": _record(aha / "manifest.json"),
                        "myocardial_roi_manifest": _record(myocardial / "manifest.json"),
                        "lv_outer_filled_manifest": None if lv_outer is None else _record(lv_outer / "manifest.json")},
            "downstream_only": True, "no_training": True, "no_reconstruction": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "case", "native-sax-root", "prepared-sax", "mapping", "roi-bundle", "proposed-bundle", "fit-first-bundle", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--outer-mask-bundle")
    args = vars(parser.parse_args())
    print(build_paired_final_pose_postprocess(
        subject=args["subject"], case=args["case"], native_sax_root=args["native_sax_root"],
        prepared_sax=args["prepared_sax"], mapping=args["mapping"], roi_bundle=args["roi_bundle"],
        proposed_bundle=args["proposed_bundle"], fit_first_bundle=args["fit_first_bundle"],
        outer_mask_bundle=args["outer_mask_bundle"], output=args["output"],
    ))


if __name__ == "__main__":
    main()
