"""Create a physical, copy-only reproducibility archive from frozen route evidence.

This intentionally does not write an archive README.  The README is generated
only after every planned copy and SHA256 record has completed successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CopyEntry:
    label: str
    source: Path
    destination: Path


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_preprocessed_subject_root(report: dict[str, Any]) -> Path:
    manifests = report.get("native_cache_manifests", {})
    source = manifests.get("sax", {}).get("source_preprocessed", {}).get("path")
    if not isinstance(source, str) or not source:
        raise ValueError("freeze report lacks native_cache_manifests.sax.source_preprocessed.path")
    path = Path(source)
    if path.name != "preprocessed.mat" or path.parent.name != "sax":
        raise ValueError(f"unexpected source preprocessed path: {path}")
    return path.parents[1]


def build_copy_plan(
    report: dict[str, Any], *, raw_dicom_subject_root: str | Path, deployment_manifest: str | Path,
) -> tuple[CopyEntry, ...]:
    """Return explicit source-to-relative-destination copies for one subject."""
    subject = report.get("subject")
    roots = report.get("roots", {})
    if not isinstance(subject, str) or not subject:
        raise ValueError("freeze report lacks subject")
    required = (
        "shared_prepared_observations", "shared_manual_roi", "shared_sparse_inputs",
        "fit_first_matlab_mapping", "fit_first_native_cache", "fit_first_reconstructions",
        "proposed_reconstructions", "final_pose_evaluation", "common_native_evaluation",
    )
    missing = [name for name in required if not isinstance(roots.get(name), str) or not roots[name]]
    if missing:
        raise ValueError(f"freeze report lacks roots: {missing}")
    input_base = Path("03_subject_examples") / subject / "01_inputs"
    output_base = Path("03_subject_examples") / subject / "02_outputs"
    return (
        CopyEntry("raw_dicom", Path(raw_dicom_subject_root), input_base / "raw_dicom"),
        CopyEntry("deployment_manifest", Path(deployment_manifest), input_base / "deployment_manifest.json"),
        CopyEntry("preprocessed", _source_preprocessed_subject_root(report), input_base / "preprocessed"),
        CopyEntry("prepared", Path(roots["shared_prepared_observations"]), input_base / "prepared"),
        CopyEntry("roi", Path(roots["shared_manual_roi"]), input_base / "roi"),
        CopyEntry("sparse_inputs", Path(roots["shared_sparse_inputs"]), input_base / "sparse_inputs"),
        CopyEntry("mapping", Path(roots["fit_first_matlab_mapping"]), output_base / "mapping"),
        CopyEntry("native_cache", Path(roots["fit_first_native_cache"]), output_base / "native_cache"),
        CopyEntry("fit_first", Path(roots["fit_first_reconstructions"]), output_base / "fit_first"),
        CopyEntry("proposed", Path(roots["proposed_reconstructions"]), output_base / "proposed"),
        CopyEntry("final_pose", Path(roots["final_pose_evaluation"]), output_base / "final_pose"),
        CopyEntry("common_native", Path(roots["common_native_evaluation"]), output_base / "common_native"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_entry(entry: CopyEntry, archive_root: Path, records: list[dict[str, str]]) -> None:
    if not entry.source.exists():
        raise FileNotFoundError(f"required archive source is missing: {entry.label}: {entry.source}")
    destination = archive_root / entry.destination
    if entry.source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.source, destination)
        records.append({"label": entry.label, "source": str(entry.source), "archive": str(entry.destination),
                        "sha256": _sha256(destination)})
        return
    shutil.copytree(entry.source, destination, copy_function=shutil.copy2)
    for copied in sorted(path for path in destination.rglob("*") if path.is_file()):
        relative = copied.relative_to(destination)
        records.append({"label": entry.label, "source": str(entry.source / relative),
                        "archive": str(copied.relative_to(archive_root)), "sha256": _sha256(copied)})


def _archive_code(code_root: Path, archive_root: Path) -> dict[str, str]:
    status = subprocess.run(["git", "-C", str(code_root), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout
    tracked = [line for line in status.splitlines() if not line.startswith("??")]
    if tracked:
        raise ValueError("refusing code snapshot with tracked modifications: " + "; ".join(tracked))
    commit = subprocess.run(["git", "-C", str(code_root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
    target = archive_root / "01_code_snapshot"
    target.mkdir(parents=True)
    subprocess.run(["git", "-C", str(code_root), "archive", "--format=tar", "--output", str(target / "source.tar"), commit], check=True)
    # ``git bundle create`` accepts a revision *range*.  A raw object ID is
    # treated as an exclusion-only range by Git and can yield an empty bundle;
    # symbolic HEAD includes the complete reachable history for this snapshot.
    subprocess.run(["git", "-C", str(code_root), "bundle", "create", str(target / "source.bundle"), "HEAD"], check=True)
    (target / "git_status.txt").write_text(status, encoding="utf-8")
    record = {"code_root": str(code_root), "commit": commit, "source_tar_sha256": _sha256(target / "source.tar"),
              "source_bundle_sha256": _sha256(target / "source.bundle")}
    (target / "COMMIT.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def materialize_archive(
    *, freeze_root: str | Path, archive_root: str | Path, code_root: str | Path,
    raw_dicom_subject_root: str | Path, deployment_manifest: str | Path,
) -> Path:
    """Copy all frozen inputs/outputs into a new immutable-layout archive."""
    freeze, archive, code = Path(freeze_root).resolve(), Path(archive_root).resolve(), Path(code_root).resolve()
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite archive root: {archive}")
    report = _load(freeze / "live_route_freeze.json")
    semantic = _load(freeze / "proposed_semantic_verification.json")
    if any(value.get("status") != "PASS" for value in semantic.get("cases", {}).values()):
        raise ValueError("refusing archive materialization: Proposed semantic verification is not fully PASS")
    plan = build_copy_plan(report, raw_dicom_subject_root=raw_dicom_subject_root, deployment_manifest=deployment_manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{archive.name}.partial-", dir=archive.parent))
    try:
        records: list[dict[str, str]] = []
        _copy_entry(CopyEntry("live_route_freeze", freeze, Path("00_live_route_freeze")), partial, records)
        code_record = _archive_code(code, partial)
        for entry in plan:
            _copy_entry(entry, partial, records)
        manifest = {
            "schema_version": "scmr_v5_multimap_reproducibility_archive/v1",
            "subject_examples": [report["subject"]],
            "freeze_root": str(freeze),
            "code_snapshot": code_record,
            "copy_plan": [{"label": item.label, "source": str(item.source), "archive": str(item.destination)} for item in plan],
            "file_count": len(records),
        }
        integrity = partial / "99_integrity"
        integrity.mkdir()
        (integrity / "archive_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (integrity / "file_sha256.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(partial, archive)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--raw-dicom-subject-root", required=True)
    parser.add_argument("--deployment-manifest", required=True)
    args = parser.parse_args()
    print(materialize_archive(**vars(args)))


if __name__ == "__main__":
    main()
