"""Run fair map-domain PSF-matched reprojection of exported T1/T2 maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .metrics import agreement_metrics
from .map_domain_psf import (
    grid_from_registered_nifti,
    grids_from_trad_prepared,
    git_commit,
    load_nifti_map,
    reproject_2dfit_exported_maps,
    reproject_trad_exported_maps,
    sha256,
)


def _registered_paths(folder: Path) -> list[Path]:
    paths = [path for path in folder.iterdir() if path.name.endswith((".nii", ".nii.gz")) and not path.name.startswith("mask_")]
    if not paths:
        raise FileNotFoundError(f"No registered NIfTI slices in {folder}")
    try:
        return sorted(paths, key=lambda path: int(path.name.split(".nii")[0]))
    except ValueError as exc:
        raise ValueError(f"Registered slice names in {folder} must use NeSVoR integer ids (for example 0.nii.gz).") from exc


def _discover_prepared(root: Path, subject_id: str) -> list[Path]:
    base = root / subject_id if (root / subject_id).is_dir() else root
    paths = [base / stack / "observations.npz" for stack in ("sax", "2ch", "4ch")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Trad --prepared-root must contain sax/2ch/4ch observations.npz; missing {missing}")
    return paths


def _write_outputs(output: Path, results: dict[str, list]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for parameter, entries in results.items():
        for index, entry in enumerate(entries):
            target = output / f"{parameter}_native_map_domain_psf_{index:03d}.npz"
            np.savez_compressed(target, values_ms=entry.values, support=entry.support)
            hashes[target.name] = sha256(target)
    return hashes


def _write_reference_comparison(
    output: Path, results: dict[str, list], *, reference_root: Path, preprocessed_root: Path,
    mask_bundle: str | None, subject_id: str,
) -> dict[str, str]:
    """Write residual/common-support arrays and requested reference-space metrics."""

    from .reference_2d import STACKS, load_verified_reference

    reference = load_verified_reference(reference_root, subject_id, preprocessed_root)
    hashes: dict[str, str] = {}
    metrics: dict[str, object] = {"global_common_support": {}, "sax_myocardium": {}}
    masks = None
    if mask_bundle:
        with np.load(Path(mask_bundle) / "sax_myocardium_masks.npz", allow_pickle=False) as data:
            masks = {key: np.asarray(data[key], dtype=bool) for key in ("myocardium_core_1px", "myocardium_full", "myocardium_core_legacy")}
    for parameter in ("T1", "T2"):
        field = "t1_ms" if parameter == "T1" else "t2_ms"
        predicted = [entry.values for entry in results[parameter]]
        support = [entry.support for entry in results[parameter]]
        expected = [getattr(reference.stacks[stack], field) for stack in STACKS]
        valid_reference = [reference.stacks[stack].valid_mask for stack in STACKS]
        if len(predicted) != sum(values.shape[0] for values in expected):
            raise ValueError("Reprojected slice count does not match verified native reference group count.")
        if any(predicted[index].shape != item.shape for index, item in enumerate(np.concatenate(expected, axis=0))):
            raise ValueError("Reprojected slice shape does not match verified native reference.")
        prediction = np.stack(predicted)
        supported = np.stack(support)
        native = np.concatenate(expected, axis=0)
        common = supported & np.concatenate(valid_reference, axis=0) & np.isfinite(prediction) & np.isfinite(native)
        residual = np.where(common, prediction - native, np.nan).astype(np.float32)
        target = output / f"{parameter}_native_map_domain_psf_comparison.npz"
        np.savez_compressed(target, predicted_ms=prediction, native_reference_ms=native, residual_ms=residual, common_support=common)
        hashes[target.name] = sha256(target)
        metrics["global_common_support"][parameter] = agreement_metrics(native, prediction, common)
        if masks is not None:
            sax_count = reference.stacks["sax"].t1_ms.shape[0]
            metrics["sax_myocardium"][parameter] = {
                key: agreement_metrics(native[:sax_count], prediction[:sax_count], common[:sax_count] & roi)
                for key, roi in masks.items()
            }
    metrics_path = output / "map_domain_psf_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    hashes[metrics_path.name] = sha256(metrics_path)
    return hashes


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    volumes = {"T1": load_nifti_map(args.t1_volume), "T2": load_nifti_map(args.t2_volume)}
    if args.method == "trad":
        grids = grids_from_trad_prepared(_discover_prepared(Path(args.prepared_root), args.subject_id), args.final_poses)
        results = reproject_trad_exported_maps(volumes, grids, n_samples=args.n_samples, seed=args.seed)
        geometry_source = str(Path(args.prepared_root).resolve())
        pose_source: str | dict[str, str] = str(Path(args.final_poses).resolve())
    else:
        folders = {"T1": Path(args.t1_registered_slices), "T2": Path(args.t2_registered_slices)}
        grids = {parameter: [grid_from_registered_nifti(path) for path in _registered_paths(folder)] for parameter, folder in folders.items()}
        results = reproject_2dfit_exported_maps(volumes, grids, n_samples=args.n_samples, seed=args.seed)
        geometry_source = {parameter: str(folder.resolve()) for parameter, folder in folders.items()}
        pose_source = geometry_source
    output_hashes = _write_outputs(output, results)
    native_reference = None
    if args.native_reference:
        reference_path = Path(args.native_reference).resolve() / "native_reference_manifest.json"
        if not reference_path.is_file():
            raise FileNotFoundError(f"Expected native reference manifest at {reference_path}")
        native_reference = {"manifest_path": str(reference_path), "manifest_sha256": sha256(reference_path)}
        output_hashes.update(_write_reference_comparison(
            output, results, reference_root=Path(args.native_reference).resolve(),
            preprocessed_root=Path(args.preprocessed_root).resolve(), mask_bundle=args.mask_bundle,
            subject_id=args.subject_id,
        ))
    manifest = {
        "schema": "map_domain_psf_reprojection/v1",
        "subject_id": args.subject_id,
        "method": args.method,
        "domain": "map",
        "bloch_used": False,
        "dictionary_used": False,
        "map_units": "ms",
        "input_volumes": {"T1": {"path": str(Path(args.t1_volume).resolve()), "sha256": sha256(args.t1_volume)}, "T2": {"path": str(Path(args.t2_volume).resolve()), "sha256": sha256(args.t2_volume)}},
        "native_reference": native_reference,
        "geometry_source": geometry_source,
        "pose_source": pose_source,
        "psf_implementation": "map_domain_psf.reproject_volume_to_native_map_psf; NeSVoR resolution2sigma Gaussian convention",
        "psf_source": "corrected 2D-fit-first evaluation/reprojection/generate_psf_matched_reprojections.py -> nesvor.svr.reconstruction.simulate_slices + get_PSF",
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "physical_resolution_thickness_source": "NativeGrid affine_ras_rc / prepared observations or final registered NIfTI",
        "output_sha256": output_hashes,
        "git_commit": git_commit(),
    }
    manifest_path = output / "map_domain_psf_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("trad", "2dfit"), required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--t1-volume", required=True)
    parser.add_argument("--t2-volume", required=True)
    parser.add_argument("--n-samples", choices=(32, 128), type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-reference", help="Optional verified native 2-D reference bundle recorded in provenance.")
    parser.add_argument("--preprocessed-root", help="Required with --native-reference for provenance validation.")
    parser.add_argument("--mask-bundle", help="Optional bundle containing sax_myocardium_masks.npz.")
    parser.add_argument("--prepared-root")
    parser.add_argument("--final-poses")
    parser.add_argument("--t1-registered-slices")
    parser.add_argument("--t2-registered-slices")
    args = parser.parse_args()
    if args.method == "trad" and (not args.prepared_root or not args.final_poses):
        parser.error("--method trad requires --prepared-root and --final-poses")
    if args.method == "2dfit" and (not args.t1_registered_slices or not args.t2_registered_slices):
        parser.error("--method 2dfit requires --t1-registered-slices and --t2-registered-slices")
    if args.native_reference and not args.preprocessed_root:
        parser.error("--native-reference requires --preprocessed-root")
    print(run(args))


if __name__ == "__main__":
    main()
