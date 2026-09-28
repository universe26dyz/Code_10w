"""Read-only G0 central-plane geometry/correspondence audit."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:  # support the required direct-script server invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quality_experiments.G0_geometry_correspondence.correspondence import apply_orientation, audit_stack
from quality_experiments.G0_geometry_correspondence.pose_audit import pose_delta_rows
from quality_experiments.G0_geometry_correspondence.signal_adapter import extract_weight_zero_signal
from quality_experiments.G0_geometry_correspondence.visualization import render_assignment, render_pose_drift, render_score_matrix, render_worst_pair
from trad.evaluation.scmr.reference_2d import STACKS, load_verified_reference
from trad.evaluation.scmr.run_scmr_fig12 import _extract_quantitative_stack


def ensure_empty_output(output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _load_signal_status(root: str | None) -> tuple[str, dict[str, object]]:
    if root is None:
        return "DEPENDENCY_PENDING", {"reason": "No D2-compatible signal artifact root supplied."}
    root_path = Path(root)
    results: dict[str, object] = {}
    for stack in STACKS:
        path = root_path / f"signal_reprojection_{stack}.npz"
        if not path.is_file():
            return "DEPENDENCY_PENDING", {"reason": f"Missing D2-compatible artifact: {path}"}
        with np.load(path, allow_pickle=False) as archive:
            selected = extract_weight_zero_signal({key: np.asarray(archive[key]) for key in archive.files})
        results[stack] = {"groups": int(selected["group_idx"].size), "weight_idx": 0, "valid_support_count": int(selected["support"].sum())}
    return "AVAILABLE", results


def _pose_rows(run_root: Path, group_counts: dict[str, int]) -> tuple[str, list[dict[str, object]]]:
    path = run_root / "final_rigid_poses.json"
    if not path.is_file():
        return "DEPENDENCY_PENDING", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"post_stack_init_axisangle_physical", "axisangle_physical"}
    if not required.issubset(payload):
        return "DEPENDENCY_PENDING", []
    initial, final = np.asarray(payload["post_stack_init_axisangle_physical"], float), np.asarray(payload["axisangle_physical"], float)
    if initial.shape != final.shape or initial.ndim != 2 or initial.shape[1] != 6:
        raise ValueError("final_rigid_poses.json has invalid physical axis-angle arrays.")
    offset, rows = 0, []
    for stack in STACKS:
        count = group_counts[stack]
        rows.extend(pose_delta_rows(stack, initial[offset:offset + count], final[offset:offset + count], group_offset=offset))
        offset += count
    if offset != final.shape[0]:
        raise ValueError("Final pose count does not match central-plane stack group counts.")
    return "AVAILABLE", rows


def run(args: argparse.Namespace) -> Path:
    output = ensure_empty_output(args.output)
    for relative in ("config", "metrics", "figures", "artifacts", "logs", "commands"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    (output / "commands" / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    reference = load_verified_reference(args.native_reference_root, args.subject_id, args.preprocessed_root)
    run_root = Path(args.run_root).resolve()
    all_rows: list[dict[str, object]] = []
    summaries: dict[str, object] = {}
    group_counts: dict[str, int] = {}
    for stack in STACKS:
        central = _extract_quantitative_stack(reference, run_root, stack)
        group_counts[stack] = int(central["support"].shape[0])
        for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
            native = np.asarray(getattr(reference.stacks[stack], key), np.float32)
            native_support = np.asarray(reference.stacks[stack].valid_mask, bool) & np.isfinite(native)
            audit = audit_stack(native, central[key], native_support, central["support"])
            rows = [{"subject_id": args.subject_id, "stack": stack, "parameter": parameter, **row} for row in audit.rows]
            all_rows.extend(rows)
            summaries[f"{parameter}_{stack}"] = {
                "diagonal_assignment_fraction": float(np.mean(audit.assignment == np.arange(audit.assignment.size))),
                "non_diagonal_count": int(np.sum(audit.assignment != np.arange(audit.assignment.size))),
                "non_identity_orientation_count": int(sum(row["best_orientation"] != "IDENTITY" for row in audit.rows)),
                "minimum_assignment_margin": float(min(float(row["assignment_margin"]) for row in audit.rows)),
                "worst_expected_group": int(min(audit.rows, key=lambda row: float(row["correlation"]))["expected_group"]),
            }
            np.savez_compressed(output / "artifacts" / f"{parameter}_{stack}_score_matrix.npz", correlation=audit.score_matrix, orientation=audit.orientation_matrix.astype("U32"), assignment=audit.assignment)
            render_score_matrix(audit.score_matrix, stack, parameter, output / "figures" / f"{parameter}_{stack}_score_matrix.png")
            render_assignment(audit.assignment, stack, parameter, output / "figures" / f"{parameter}_{stack}_assignment.png")
            worst = min(audit.rows, key=lambda row: float(row["correlation"]))
            expected, matched, orientation = int(worst["expected_group"]), int(worst["content_matched_group"]), str(worst["best_orientation"])
            oriented = apply_orientation(central[key][matched], orientation)
            if oriented.shape == native[expected].shape:
                render_worst_pair(native[expected], oriented, stack, parameter, expected, matched, orientation, output / "figures" / f"{parameter}_{stack}_worst_pair.png")
    signal_status, signal_detail = _load_signal_status(args.signal_artifacts_root)
    pose_status, poses = _pose_rows(run_root, group_counts)
    if poses:
        _csv(output / "metrics" / "G0_pose_drift.csv", poses)
        render_pose_drift(poses, output / "figures" / "pose_drift.png")
    _csv(output / "metrics" / "G0_per_slice.csv", all_rows)
    _json(output / "metrics" / "G0_summary.json", {"schema": "code10w_g0_geometry_audit/v1", "official_pairing": "native group g <-> reconstruction group g; Hungarian result is diagnostic only", "summaries": summaries, "signal_status": signal_status, "pose_status": pose_status})
    _json(output / "artifacts" / "G0_signal_adapter.json", {"status": signal_status, "detail": signal_detail})
    _json(output / "config" / "audit_args.json", vars(args))
    manifest = {"schema": "code10w_g0_geometry_audit/v1", "implementation_revision": "G0_geometry_correspondence_baseline_v2", "experiment_id": "G0_geometry_correspondence_baseline_v1", "status": "COMPLETE", "subject_id": args.subject_id, "git_commit": _git_commit(), "baseline_read_only": True, "primary_domain": "central/no-PSF quantitative maps", "signal_status": signal_status, "pose_status": pose_status, "input_provenance": {"run_root": str(run_root), "native_reference_manifest": str(reference.manifest_path), "native_reference_hashes": reference.file_hashes}}
    _json(output / "manifest.json", manifest)
    (output / "README.md").write_text("# G0 geometry/correspondence audit\n\nHungarian content matching is diagnostic only. Official metrics retain native group `g` paired with reconstructed group `g`.\n", encoding="utf-8")
    (output / "RESULT_SUMMARY.md").write_text("# G0 result summary\n\nFormal G0-map result generated from the supplied central/no-PSF artifacts. Signal status: " + signal_status + ". Pose status: " + pose_status + ". See `metrics/G0_summary.json` for actual measurements.\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--run-root", required=True, help="FrozenMLP baseline run containing central-plane exports and optional final poses.")
    parser.add_argument("--native-reference-root", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--signal-artifacts-root", help="Optional D2-compatible signal artifact root; absent means DEPENDENCY_PENDING.")
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
