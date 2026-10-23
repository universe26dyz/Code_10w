"""Apply an explicitly selected sparse profile to one standardized subject."""
from __future__ import annotations

import argparse
from pathlib import Path

from .final_method_selection import validate_final_method_selection
from .sparse import build_sparse_profile_condition
from .standardize_subject_slices import validate_standardization_manifest


def materialize_final_subject_profile(*, final_method_selection: str | Path, standardization_manifest: str | Path, subject: str, output_root: str | Path, fold: int | None = None, dry_run: bool = False) -> dict:
    selection = validate_final_method_selection(final_method_selection)
    standardization = validate_standardization_manifest(standardization_manifest, subject=subject)
    policy = selection["profile_replication_policy"]
    if policy["mode"] == "one_preregistered_fold":
        if fold not in (None, policy["fold_id"]):
            raise ValueError("final selection fixes one preregistered profile fold")
        fold = policy["fold_id"]
    elif fold is None:
        raise ValueError("all_profile_folds requires an explicit fold for each materialization")
    profile = selection["selected_profile"]
    standardized_subject_root = Path(standardization["resolved_source_root"]).parent / subject
    # The manifest records the output path, which is the authoritative standardized source.
    standardized_subject_root = Path(standardization["stacks"]["sax"]["output_observation"]["path"]).resolve().parents[1]
    result = build_sparse_profile_condition(standardized_subject_root.parent, subject, profile["profile_path"], fold, output_root, dry_run=dry_run)
    if result.get("profile_id") != profile["profile_id"]:
        raise ValueError("materialized profile identity disagrees with final selection")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-method-selection", required=True); parser.add_argument("--standardization-manifest", required=True); parser.add_argument("--subject", required=True); parser.add_argument("--output-root", required=True); parser.add_argument("--fold", type=int); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); print(materialize_final_subject_profile(final_method_selection=args.final_method_selection, standardization_manifest=args.standardization_manifest, subject=args.subject, output_root=args.output_root, fold=args.fold, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
