"""Read-only Final5 publication-visualization consumer.

This module consumes validated central endpoints and the immutable Final5 v2
metric bundle.  It creates no quantitative metric and never runs training,
reconstruction, endpoint export, or dictionary matching.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trad.evaluation.scmr.quality_control import DISPLAY_RANGES_MS, load_legacy_scmr_colorbars

from .contracts import sha256, write_immutable_json
from .evaluation import reproject_volume_to_native_plane
from .final_cyj_reporting import _load_all_verified_native_cache_groups
from .final_native_grid_metrics import native_ras_affine
from .final_sparse_native_metrics import validate_sparse_endpoint_roots
from .visualization import native_panel_specs


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = {"t1": "T1", "t2": "T2"}
METRIC_SCHEMA = "final5_native_grid_metric_bundle/v2"


def require_new_publication_output(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _published_record(staging_path: Path, published_path: Path) -> dict[str, str]:
    return {"path": str(published_path.resolve()), "sha256": sha256(staging_path)}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def publication_group_roles(sparse: Mapping[str, Any], source_groups: Mapping[str, list[int]]) -> dict[tuple[str, int], str]:
    """Classify all source planes only from the declared sparse manifest."""
    if sparse.get("case_id") == "L100":
        return {(stack, int(group)): "observed" for stack in STACKS for group in source_groups[stack]}
    stacks = sparse.get("stacks")
    if not isinstance(stacks, Mapping):
        raise ValueError("sparse manifest lacks stacks")
    roles = {("sax", int(group)): "observed" for group in source_groups["sax"]}
    for stack in ("2ch", "4ch"):
        item = stacks.get(stack)
        if not isinstance(item, Mapping):
            raise ValueError(f"sparse manifest lacks {stack}")
        retained, heldout = item.get("retained_original_group_ids"), item.get("heldout_original_group_ids")
        if not isinstance(retained, list) or not isinstance(heldout, list) or any(not isinstance(value, int) or isinstance(value, bool) for value in [*retained, *heldout]):
            raise ValueError(f"{stack}: retained/heldout IDs must be integer lists")
        if set(retained) & set(heldout) or set(retained) | set(heldout) != set(source_groups[stack]):
            raise ValueError(f"{stack}: manifest retained/heldout IDs do not partition native source groups")
        roles.update({(stack, int(group)): "retained" for group in retained})
        roles.update({(stack, int(group)): "heldout" for group in heldout})
    return roles


def display_support(source_support: np.ndarray, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray, proposed_valid: np.ndarray, fit_first_valid: np.ndarray) -> np.ndarray:
    """Foreground display support; deliberately not the formal metric ROI."""
    return (np.asarray(source_support, bool) & np.isfinite(reference) & np.asarray(proposed_valid, bool)
            & np.asarray(fit_first_valid, bool) & np.isfinite(proposed) & np.isfinite(fit_first))


def crop_bounds(support: np.ndarray, *, pad: int = 3) -> dict[str, int]:
    rows, columns = np.where(np.asarray(support, bool))
    if not len(rows):
        raise ValueError("empty common foreground display support")
    return {"row_start": max(0, int(rows.min()) - pad), "row_stop": min(support.shape[0], int(rows.max()) + pad + 1), "column_start": max(0, int(columns.min()) - pad), "column_stop": min(support.shape[1], int(columns.max()) + pad + 1)}


def _crop(values: np.ndarray, bounds: Mapping[str, int]) -> np.ndarray:
    return np.asarray(values)[int(bounds["row_start"]):int(bounds["row_stop"]), int(bounds["column_start"]):int(bounds["column_stop"])]


def _save_panel(path: Path, values: np.ndarray, support: np.ndarray, *, cmap: Any, limits: tuple[float, float]) -> None:
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(4, 4), frameon=False)
    axis = figure.add_axes((0, 0, 1, 1)); axis.imshow(np.ma.masked_where(~support, values), cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
    axis.set_axis_off(); figure.savefig(path, dpi=240, transparent=True, pad_inches=0)
    figure.savefig(path.with_suffix(".pdf"), transparent=True, pad_inches=0)
    plt.close(figure)


def render_single_element_plane(output: str | Path, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray, support: np.ndarray, *, parameter: str, bounds: Mapping[str, int]) -> dict[str, Path]:
    """Save five CYJ-style panels sharing one crop and parameter LUT contract."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    parameter_lut = native_panel_specs(parameter)[0]
    mapping, limits = parameter_lut["cmap"], parameter_lut["limits"]
    maps = {"native_reference": reference, "proposed": proposed, "fit_first": fit_first,
            "proposed_absolute_error": np.abs(np.asarray(proposed) - np.asarray(reference)),
            "fit_first_absolute_error": np.abs(np.asarray(fit_first) - np.asarray(reference))}
    result: dict[str, Path] = {}
    for name, values in maps.items():
        target = output / f"{name}.png"; result[name] = target
        _save_panel(target, _crop(values, bounds), _crop(support, bounds), cmap=mapping,
                    limits=limits)
    return result


def _save_colorbar(path: Path, *, cmap: Any, limits: tuple[float, float], label: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize
    figure = plt.figure(figsize=(1.0, 4.0)); axis = figure.add_axes((0.30, 0.04, 0.28, 0.92))
    ColorbarBase(axis, cmap=cmap, norm=Normalize(*limits), orientation="vertical", label=label)
    figure.savefig(path, dpi=240, transparent=True); figure.savefig(path.with_suffix(".pdf"), transparent=True); plt.close(figure)


def _paths(root: Path, subject: str, case_id: str) -> dict[str, Path]:
    central = root / "05_formal_endpoints" / "central"
    return {"sparse": root / "03_sparse_inputs" / subject / case_id / "sparse_manifest.json", "native": root / "02_standardized_12_15_10" / "native_dictionary" / subject,
            "proposed": central / "proposed" / subject / case_id, "fit_first": central / "fit_first" / subject / case_id,
            "metric": root / "06_metrics" / "final_native_grid_v2" / subject / case_id}


def _validated_inputs(root: Path, subject: str, case_id: str) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = _paths(root, subject, case_id)
    sparse_path, metric_path = paths["sparse"], paths["metric"] / "manifest.json"
    sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    if sparse.get("subject") != subject or sparse.get("case_id") != case_id or sparse.get("status") != "ELIGIBLE":
        raise ValueError("sparse manifest subject/case/status mismatch")
    if (metric.get("schema"), metric.get("subject"), metric.get("condition"), metric.get("endpoint")) != (METRIC_SCHEMA, subject, case_id, "central"):
        raise ValueError("Final5 v2 metric schema/identity/endpoint mismatch")
    if metric.get("sparse_manifest") != _record(sparse_path):
        raise ValueError("Final5 v2 metric sparse-manifest provenance mismatch")
    volumes, endpoint = validate_sparse_endpoint_roots(paths["proposed"], paths["fit_first"], case_id, subject=subject)
    provenance = metric.get("endpoint_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("proposed_manifest") != endpoint["proposed_manifest"] or provenance.get("fit_first_manifest") != endpoint["fit_first_manifest"]:
        raise ValueError("Final5 v2 metric endpoint provenance mismatch")
    return paths, sparse, metric, volumes


def publication_readiness(*, final5_root: str | Path, subject: str, case_id: str, output: str | Path) -> dict[str, Any]:
    """Read-only admission check; it never creates output directories."""
    output = Path(output)
    if output.exists():
        return {"status": "ALREADY_RENDERED", "output": str(output)}
    try:
        paths, sparse, metric, _volumes = _validated_inputs(Path(final5_root), subject, case_id)
        caches = {stack: _load_all_verified_native_cache_groups(paths["native"], paths["sparse"], stack) for stack in STACKS}
        roles = publication_group_roles(sparse, {stack: [int(group) for group in caches[stack]["group_idx"]] for stack in STACKS})
    except FileNotFoundError as exc:
        return {"status": "MISSING_INPUT", "reason": str(exc)}
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED_INVALID", "reason": str(exc)}
    return {"status": "READY", "subject": subject, "case_id": case_id, "metric_schema": metric["schema"], "groups": {f"{stack}/{group}": role for (stack, group), role in roles.items()}}


def render_final5_publication_visualization(*, final5_root: str | Path, subject: str, case_id: str, output: str | Path, parameter: str | None = None, stack: str | None = None, group: int | None = None, dry_run: bool = False) -> Path | dict[str, Any]:
    """Render clean individual native-plane assets from immutable Final5 v2 inputs."""
    output = Path(output); readiness = publication_readiness(final5_root=final5_root, subject=subject, case_id=case_id, output=output)
    if dry_run:
        return readiness
    if readiness["status"] != "READY":
        raise ValueError(f"publication visualization is not ready: {readiness}")
    require_new_publication_output(output)
    import nibabel as nib
    paths, sparse, metric, volumes = _validated_inputs(Path(final5_root), subject, case_id)
    images = {name: {method: nib.load(str(path)) for method, path in entries.items()} for name, entries in volumes.items()}
    caches = {item: _load_all_verified_native_cache_groups(paths["native"], paths["sparse"], item) for item in STACKS}
    roles = publication_group_roles(sparse, {item: [int(value) for value in caches[item]["group_idx"]] for item in STACKS})
    selected_parameters = (parameter.lower(),) if parameter else tuple(PARAMETERS)
    selected_stacks = (stack.lower(),) if stack else STACKS
    if set(selected_parameters).difference(PARAMETERS) or set(selected_stacks).difference(STACKS):
        raise ValueError("unknown parameter or stack filter")
    output.parent.mkdir(parents=True, exist_ok=True); staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        bars = load_legacy_scmr_colorbars()
        assets = staging / "assets"; assets.mkdir()
        for key, display in PARAMETERS.items():
            _save_colorbar(assets / f"{display}_map_colorbar.png", cmap=bars.mapping[display], limits=DISPLAY_RANGES_MS[display], label=f"{display} (ms)")
        records: list[dict[str, Any]] = []
        for stack_name in selected_stacks:
            cache = caches[stack_name]
            for index, value in enumerate(cache["group_idx"]):
                original_group = int(value)
                if group is not None and original_group != group:
                    continue
                affine = native_ras_affine(cache["affine_lps_rc"][index]); role = roles[(stack_name, original_group)]
                for key in selected_parameters:
                    display = PARAMETERS[key]; reference = np.asarray(cache[f"{key}_ms"][index], float); foreground = np.asarray(cache["source_support"][index], bool)
                    proposed, pvalid = reproject_volume_to_native_plane(images[key]["proposed"].get_fdata(), images[key]["proposed"].affine, affine, reference.shape)
                    fit, fvalid = reproject_volume_to_native_plane(images[key]["fit_first"].get_fdata(), images[key]["fit_first"].affine, affine, reference.shape)
                    support = display_support(foreground, reference, proposed, fit, pvalid, fvalid); bounds = crop_bounds(support)
                    group_root = staging / subject / case_id / display / stack_name / role / f"group{original_group:03d}"
                    rendered = render_single_element_plane(group_root, reference, proposed, fit, support, parameter=display, bounds=bounds)
                    formal_mask = paths["metric"] / "native_masks" / key / stack_name / f"group{original_group:03d}.npz"
                    formal_provenance: dict[str, Any] = {"metric_manifest": _record(paths["metric"] / "manifest.json"), "native_mask": None}
                    if formal_mask.is_file():
                        with np.load(formal_mask, allow_pickle=False) as stored:
                            if not np.array_equal(np.asarray(stored["native_affine_ras"], float), affine):
                                raise ValueError("formal metric native affine mismatch")
                        formal_provenance["native_mask"] = _record(formal_mask)
                    metadata = {"subject": subject, "case_id": case_id, "profile_id": sparse.get("profile_id"), "parameter": display, "stack": stack_name, "original_group_idx": original_group, "sampling_role": role, "native_affine_ras": affine.tolist(), "crop_bounds": bounds, "display_support_definition": "source_support AND finite(reference, proposed, fit_first) AND proposed_reprojection_valid AND fit_first_volume_geometry_valid; no connected-component filtering; not formal metric ROI", "formal_metric_support_provenance": formal_provenance, "proposed_endpoint_manifest": metric["endpoint_provenance"]["proposed_manifest"], "fit_first_endpoint_manifest": metric["endpoint_provenance"]["fit_first_manifest"], "sparse_manifest": _record(paths["sparse"]), "native_reference_provenance": cache["provenance"], "display_lut": bars.metadata["mapping_colormaps"][display], "display_range_ms": list(DISPLAY_RANGES_MS[display]), "error_range_ms": list(DISPLAY_RANGES_MS[display]), "error_display": "absolute_error_ms uses the same parameter LUT/range; no residual colorbar", "interpolation": {"maps": "linear order=1", "binary_masks": "nearest-neighbour order=0"}, "clean_single_element_files": {name: path.name for name, path in rendered.items()}, "git_sha": _git_sha()}
                    write_immutable_json(group_root / "metadata.json", metadata); records.append(metadata)
        write_immutable_json(staging / "publication_visualization_manifest.json", {"schema": "final5_publication_single_element_visualization/v1", "subject": subject, "case_id": case_id, "publication_consumer_only": True, "formal_metric_authority": str(paths["metric"].resolve()), "formal_metric_schema": METRIC_SCHEMA, "no_training": True, "no_reconstruction": True, "no_metric_recomputation": True, "records": records, "assets": {name: _published_record(path, output / "assets" / path.name) for name, path in {item.name: item for item in assets.glob("*.png")}.items()}})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final5-root", required=True); parser.add_argument("--subject", required=True); parser.add_argument("--case", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--parameter", choices=("t1", "t2")); parser.add_argument("--stack", choices=STACKS); parser.add_argument("--group", type=int); parser.add_argument("--dry-run", action="store_true")
    args = vars(parser.parse_args()); print(render_final5_publication_visualization(final5_root=args["final5_root"], subject=args["subject"], case_id=args["case"], output=args["output"], parameter=args["parameter"], stack=args["stack"], group=args["group"], dry_run=args["dry_run"]))


if __name__ == "__main__":
    main()
