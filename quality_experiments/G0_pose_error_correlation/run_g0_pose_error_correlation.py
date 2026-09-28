"""Read-only exploratory association between G0 v2 pose drift and central-map error."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quality_experiments.G0_pose_error_correlation.analysis import ERROR_METRICS, POSE_METRICS, bh_fdr, join_pose_map_rows, leave_one_out_sensitivity, primary_statistics, read_csv_rows, sha256, validate_bundle
from quality_experiments.G0_pose_error_correlation.figures import correlation_grid, group_overview


EXPERIMENT_ID = "G0_POSE_ERROR_CORRELATION_CYJ_V1"
EXPECTED_STACKS = ("sax", "2ch", "4ch")
EXPECTED_PARAMETERS = ("T1", "T2")


def discover_local_v2_bundles(search_roots: list[str | Path]) -> list[Path]:
    """Find only complete corrected bundles below explicitly supplied local roots."""

    found: list[Path] = []
    for root in search_roots:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            continue
        for per_slice in root_path.rglob("G0_per_slice.csv"):
            candidate = per_slice.parent.parent
            try:
                validate_bundle(candidate)
            except (FileNotFoundError, ValueError):
                continue
            found.append(candidate.resolve())
    return sorted(set(found))


def _ensure_empty_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _summarize(stats: list[dict[str, object]], loo: list[dict[str, object]]) -> str:
    lines = ["# G0 pose drift vs central-map error correlation", "", "This is an exploratory, single-subject, small-N observational post-hoc association analysis. It cannot establish causality or determine whether Stage-B movement is correct motion correction versus objective-driven drift.", ""]
    loo_index = {(row["stack"], row["parameter"], row["pose_metric"], row["error_metric"]): row for row in loo}
    for stat in stats:
        key = (stat["stack"], stat["parameter"], stat["pose_metric"], stat["error_metric"])
        sensitivity = loo_index[key]
        lines.append(f"- {stat['parameter']} {str(stat['stack']).upper()}: {stat['pose_metric']} vs {stat['error_metric']}; N={stat['N']}, Spearman rho={stat['spearman_rho']:.4g} (raw p={stat['spearman_p_two_sided']:.4g}, BH q={stat['spearman_q_bh']:.4g}), Pearson r={stat['pearson_r']:.4g} (p={stat['pearson_p_two_sided']:.4g}); LOO rho range [{sensitivity['loo_rho_min']:.4g}, {sensitivity['loo_rho_max']:.4g}], most influential local/global group={sensitivity['most_influential_removed_group_idx']}/{sensitivity['most_influential_removed_global_group_idx']}.")
    lines.extend(["", "Interpretation: inspect direction, Pearson/Spearman agreement, LOO range, and group-level plots together. No causal or p-value-only conclusion is made by this diagnostic."])
    return "\n".join(lines) + "\n"


def validate_analysis_design(by_pair: dict[tuple[str, str], list[dict[str, object]]]) -> None:
    """Require the six G0 v2 stack/parameter associations used by the figures."""

    expected = {(stack, parameter) for stack in EXPECTED_STACKS for parameter in EXPECTED_PARAMETERS}
    observed = set(by_pair)
    if observed != expected:
        raise ValueError(f"Expected exactly the six G0 stack/parameter pairs {sorted(expected)}, got {sorted(observed)}.")


def run(input_root: str | Path, output_root: str | Path) -> Path:
    input_path, manifest, per_slice, pose = validate_bundle(input_root)
    subject_id = manifest.get("subject_id")
    if subject_id != "CYJ":
        raise ValueError(f"Expected CYJ G0 v2 bundle, got subject_id={subject_id!r}")
    joined = join_pose_map_rows(read_csv_rows(per_slice), read_csv_rows(pose))
    by_pair: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in joined:
        by_pair[(str(row["stack"]), str(row["parameter"]))].append(row)
    validate_analysis_design(by_pair)
    statistics, sensitivity = [], []
    for rows in by_pair.values():
        statistics.extend(primary_statistics(rows))
        for pose_metric in POSE_METRICS:
            for error_metric in ERROR_METRICS:
                sensitivity.append({"stack": rows[0]["stack"], "parameter": rows[0]["parameter"], "pose_metric": pose_metric, "error_metric": error_metric, **leave_one_out_sensitivity(rows, pose_metric, error_metric)})
    q_values = bh_fdr([float(row["spearman_p_two_sided"]) for row in statistics])
    for row, q_value in zip(statistics, q_values):
        row["spearman_q_bh"] = q_value
    output = Path(output_root).expanduser().resolve(); _ensure_empty_output(output)
    for relative in ("config", "metrics", "figures", "logs"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    _write_csv(output / "metrics/joined_pose_map_metrics.csv", joined)
    _write_csv(output / "metrics/correlation_statistics.csv", statistics)
    _write_csv(output / "metrics/leave_one_out_sensitivity.csv", sensitivity)
    summary_rows = [{"stack": stack, "parameter": parameter, "N": len(rows)} for (stack, parameter), rows in sorted(by_pair.items())]
    _write_csv(output / "metrics/stack_summary.csv", summary_rows)
    correlation_grid(joined, statistics, "translation_magnitude_mm", "relative_rmse", output / "figures/01_translation_vs_relative_rmse.png")
    correlation_grid(joined, statistics, "rotation_magnitude_deg", "relative_rmse", output / "figures/02_rotation_vs_relative_rmse.png")
    correlation_grid(joined, statistics, "translation_magnitude_mm", "one_minus_correlation", output / "figures/03_translation_vs_one_minus_correlation.png")
    correlation_grid(joined, statistics, "rotation_magnitude_deg", "one_minus_correlation", output / "figures/04_rotation_vs_one_minus_correlation.png")
    group_overview(joined, output / "figures/05_pose_vs_error_group_overview.png")
    config = {"experiment_id": EXPERIMENT_ID, "join_key": "stack + expected_group == group_idx", "official_pair_required": "content_matched_group == expected_group AND best_orientation == IDENTITY", "primary_tests": "Pearson and Spearman two-sided; Spearman p values BH-FDR corrected across all primary tests", "leave_one_out": "remove each local group once; select maximal absolute change from full Spearman rho", "analysis_read_only": True}
    (output / "config/analysis_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "logs/run.log").write_text("G0 pose-error correlation analysis completed from a local verified G0 v2 bundle.\n", encoding="utf-8")
    output_manifest = {"experiment_id": EXPERIMENT_ID, "subject_id": subject_id, "analysis_git_commit": _git_commit(), "input_g0_v2_root": str(input_path), "input_manifest_path": str(input_path / "manifest.json"), "input_manifest_git_commit": manifest.get("git_commit"), "g0_per_slice_sha256": sha256(per_slice), "g0_pose_drift_sha256": sha256(pose), "analysis_read_only": True, "reconstruction_rerun": False, "d2_started": False}
    (output / "manifest.json").write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# G0 pose-error correlation\n\nRead-only exploratory downstream analysis of verified G0 v2 outputs.\n", encoding="utf-8")
    (output / "RESULT_SUMMARY.md").write_text(_summarize(statistics, sensitivity), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", help="Verified local G0_geometry_correspondence_baseline_v2 bundle.")
    parser.add_argument("--search-root", action="append", help="Local root to search for a complete G0 v2 bundle; may be repeated.")
    parser.add_argument("--output-root", default="/home/universe/SVR/multimap_postprogramming/Code_10w_local_results/G0_pose_error_correlation_CYJ_v1")
    args = parser.parse_args()
    if args.input_root:
        input_root = Path(args.input_root)
    else:
        roots = args.search_root or [Path.cwd().parent]
        candidates = discover_local_v2_bundles(roots)
        if not candidates:
            print("FORMAL_LOCAL_ANALYSIS = NOT_RUN_MISSING_G0_V2_ARTIFACTS")
            return
        if len(candidates) != 1:
            parser.error("Multiple local verified G0 v2 bundles found; pass --input-root explicitly.")
        input_root = candidates[0]
    print(run(input_root, args.output_root))


if __name__ == "__main__":
    main()
