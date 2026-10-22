"""Build immutable, subject-general figure configurations from a formal grid.

The grid is intentionally declarative: each method root contains
``SUBJECT/CONDITION/ENDPOINT`` exports.  Proposed exports provide
``T1_3D.nii.gz``/``T2_3D.nii.gz``; Fit-first exports provide
``t1/volume_ms.nii.gz``/``t2/volume_ms.nii.gz``.  L100 native bundles live
below each corresponding case directory as ``native_bundle``.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from .contracts import write_immutable_json
from .final_cyj_reporting import ENDPOINTS, EXPECTED_FOLDS, METHOD_DISPLAY_ORDER, PROPOSED_METHODS


COMMON_SUPPORT_CONTRACT = "frozen_L100_display_mask AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)"


def _cases(custom_cases: list[Mapping[str, str]] | None = None) -> tuple[tuple[str, str | None], ...]:
    legacy = tuple((f"{level}_fold{fold}", None) for level, folds in EXPECTED_FOLDS.items() for fold in folds)
    if custom_cases is None:
        return legacy
    custom: list[tuple[str, str | None]] = []
    for item in custom_cases:
        profile_id, case_id = item.get("profile_id"), item.get("case_id")
        if not isinstance(profile_id, str) or not profile_id or not isinstance(case_id, str) or not case_id:
            raise ValueError("each custom case requires explicit profile_id and case_id")
        custom.append((case_id, profile_id))
    if len({case for case, _profile in custom}) != len(custom):
        raise ValueError("custom case_id values must be unique")
    return tuple(custom)


def _case_root(root: Path, subject: str, condition: str, endpoint: str) -> Path:
    return root / subject / condition / endpoint


def _methods(method_roots: Mapping[str, str | Path], subject: str, condition: str, endpoint: str) -> dict[str, dict[str, str]]:
    names = set(method_roots)
    unknown = names.difference(METHOD_DISPLAY_ORDER)
    if unknown:
        raise ValueError(f"unsupported reporting method(s): {', '.join(sorted(unknown))}")
    if "Fit-first" not in names or not names.intersection(PROPOSED_METHODS):
        raise ValueError("method_roots must include Fit-first plus at least one supported Proposed method")
    result: dict[str, dict[str, str]] = {}
    for method in METHOD_DISPLAY_ORDER:
        if method not in method_roots:
            continue
        root = _case_root(Path(method_roots[method]), subject, condition, endpoint)
        if method == "Fit-first":
            maps = {"t1": str((root / "t1" / "volume_ms.nii.gz").resolve()), "t2": str((root / "t2" / "volume_ms.nii.gz").resolve())}
            if condition == "L100_fold0":
                maps["fit_first_native_bundle"] = str((root / "native_bundle").resolve())
        else:
            maps = {"t1": str((root / "T1_3D.nii.gz").resolve()), "t2": str((root / "T2_3D.nii.gz").resolve())}
            if condition == "L100_fold0":
                maps["native_bundle"] = str((root / "native_bundle").resolve())
        result[method] = maps
    return result


def build_final_subject_figure_configs(*, subject: str, formal_grid_root: str | Path, sparse_root: str | Path, r0_native_maps_root: str | Path, native_mask_root: str | Path, method_roots: Mapping[str, str | Path], output: str | Path, custom_cases: list[Mapping[str, str]] | None = None) -> Path:
    """Write one complete config per formal condition/endpoint without server work."""
    if not isinstance(subject, str) or not subject:
        raise ValueError("subject must be explicit and non-empty")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    formal_grid_root, sparse_root = Path(formal_grid_root), Path(sparse_root)
    r0_native_maps_root, native_mask_root = Path(r0_native_maps_root), Path(native_mask_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for condition, profile_id in _cases(custom_cases):
            for endpoint in ENDPOINTS:
                case_formal = _case_root(formal_grid_root, subject, condition, endpoint)
                sparse_manifest = sparse_root / subject / condition / "sparse_manifest.json"
                payload = {
                    "schema": "phase4r_final_subject_figure_config/v2",
                    "subject": subject, "condition": condition, "endpoint": endpoint,
                    "methods": _methods(method_roots, subject, condition, endpoint),
                    "l100_masks": {"t1": str((case_formal / "l100_masks" / "T1_mask.nii.gz").resolve()), "t2": str((case_formal / "l100_masks" / "T2_mask.nii.gz").resolve())},
                    "longitudinal": {"axis": 1, "index": "support_median", "common_support_contract": COMMON_SUPPORT_CONTRACT},
                    "native_mask_root": str((native_mask_root / subject / condition).resolve()),
                    "native": {"mode": "all_source_groups", "stacks": ["sax", "2ch", "4ch"], "sparse_manifest": str(sparse_manifest.resolve()), "r0_native_maps_root": str((r0_native_maps_root / subject).resolve())},
                    "scmr_longitudinal": {"sparse_manifest": str(sparse_manifest.resolve()), "r0_native_maps_root": str((r0_native_maps_root / subject).resolve()), "reconstruction_spacing_mm": 1.0, "common_support_contract": "source comparison domain AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)"},
                }
                if profile_id is not None:
                    payload["profile_id"] = profile_id; payload["case_id"] = condition
                write_immutable_json(staging / condition / f"{endpoint}.json", payload)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--formal-grid-root", required=True)
    parser.add_argument("--sparse-root", required=True); parser.add_argument("--r0-native-maps-root", required=True)
    parser.add_argument("--native-mask-root", required=True); parser.add_argument("--method-root", action="append", required=True, metavar="METHOD=ROOT")
    parser.add_argument("--custom-case", action="append", default=[], metavar="PROFILE_ID=CASE_ID")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); methods: dict[str, str] = {}
    for item in args.method_root:
        method, separator, root = item.partition("=")
        if not separator or not method or not root or method in methods:
            raise ValueError("--method-root must be a unique METHOD=ROOT")
        methods[method] = root
    custom_cases: list[dict[str, str]] = []
    for item in args.custom_case:
        profile_id, separator, case_id = item.partition("=")
        if not separator or not profile_id or not case_id:
            parser.error("--custom-case must be PROFILE_ID=CASE_ID")
        custom_cases.append({"profile_id": profile_id, "case_id": case_id})
    print(build_final_subject_figure_configs(subject=args.subject, formal_grid_root=args.formal_grid_root, sparse_root=args.sparse_root, r0_native_maps_root=args.r0_native_maps_root, native_mask_root=args.native_mask_root, method_roots=methods, output=args.output, custom_cases=custom_cases or None))


if __name__ == "__main__":
    main()
