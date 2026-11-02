"""Add a separately hashed runtime-configuration layer to a completed archive.

The original archive is immutable evidence of the successful live route.  This
tool only adds ``02_runtime_configuration`` and refuses to replace it.  It
separates generic reusable files from subject-specific example provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


COMMON_FILES = (
    Path("mlp/configs/protocol_scmr_v5_10_100.yaml"),
    Path("mlp/configs/recon_server_train_scmr_v5_generic.yaml"),
    Path("configs/subject_stack_manifest.example.json"),
)
GENERIC_SPARSE_PROFILES = (
    Path("paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH0_4CH0.json"),
    Path("paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH0_4CH100.json"),
    Path("paper_experiments/joint_vs_2dfit/configs/sparse_profiles/SAX100_2CH100_4CH0.json"),
    Path("paper_experiments/joint_vs_2dfit/configs/sparse_profiles/S2CH100_4CH50.json"),
    Path("paper_experiments/joint_vs_2dfit/configs/sparse_profiles/S2CH50_4CH100.json"),
)
LAUNCHERS = (
    Path("paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/server_preprocess_and_prepare.sh"),
    Path("paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh"),
    Path("paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh"),
    Path("paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh"),
    Path("paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh"),
    Path("paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh"),
)
PROPOSED_PROVENANCE = ("config_resolved.yaml", "experiment_manifest.json", "run_provenance.json")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy(source: Path, destination: Path, root: Path, label: str, records: list[dict[str, str]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required runtime configuration is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append({"label": label, "source": str(source), "archive": str(destination.relative_to(root)), "sha256": _sha256(destination)})


def _code_commit(code: Path) -> str:
    status = subprocess.run(["git", "-C", str(code), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout
    tracked_changes = [line for line in status.splitlines() if not line.startswith("??")]
    if tracked_changes:
        raise ValueError("refusing runtime configuration from tracked modified code: " + "; ".join(tracked_changes))
    return subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


def materialize_runtime_configuration(archive_root: str | Path, code_root: str | Path) -> Path:
    archive, code = Path(archive_root).resolve(), Path(code_root).resolve()
    code_commit = _code_commit(code)
    manifest = _load(archive / "99_integrity" / "archive_manifest.json")
    if manifest.get("schema_version") != "scmr_v5_multimap_reproducibility_archive/v1":
        raise ValueError("not a completed SCMR-v5 reproducibility archive")
    subjects = manifest.get("subject_examples")
    if not isinstance(subjects, list) or len(subjects) != 1 or not isinstance(subjects[0], str):
        raise ValueError("runtime configuration requires exactly one subject example")
    subject = subjects[0]
    target = archive / "02_runtime_configuration"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite runtime configuration: {target}")
    partial = Path(tempfile.mkdtemp(prefix=".02_runtime_configuration.partial-", dir=archive))
    try:
        records: list[dict[str, str]] = []
        for relative in COMMON_FILES:
            _copy(code / relative, partial / "00_common" / relative.name, partial, "common", records)
        for relative in GENERIC_SPARSE_PROFILES:
            _copy(code / relative, partial / "00_common" / "sparse_profiles" / relative.name, partial, "common", records)
        for relative in LAUNCHERS:
            _copy(code / relative, partial / "01_launchers" / relative.name, partial, "launcher", records)

        example_root = archive / "03_subject_examples" / subject
        _copy(example_root / "01_inputs" / "deployment_manifest.json", partial / "02_subject_examples" / subject / "deployment_manifest.json", partial, "subject_example", records)
        sparse_root = example_root / "01_inputs" / "sparse_inputs"
        sparse_manifests = sorted(sparse_root.rglob("sparse_manifest.json"))
        if not sparse_manifests:
            raise FileNotFoundError(f"archive lacks sparse_manifest.json below: {sparse_root}")
        for source in sparse_manifests:
            _copy(source, partial / "02_subject_examples" / subject / "sparse_inputs" / source.relative_to(sparse_root), partial, "subject_example", records)
        proposed_root = example_root / "02_outputs" / "proposed"
        for name in PROPOSED_PROVENANCE:
            sources = sorted(proposed_root.rglob(name))
            if not sources:
                raise FileNotFoundError(f"archive lacks {name} below: {proposed_root}")
            for source in sources:
                _copy(source, partial / "02_subject_examples" / subject / "proposed" / source.relative_to(proposed_root), partial, "subject_example", records)

        (partial / "runtime_configuration_manifest.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8"
        )
        (partial / "runtime_configuration_metadata.json").write_text(json.dumps({
            "schema_version": "scmr_v5_runtime_configuration/v1",
            "code_root": str(code),
            "code_commit": code_commit,
            "file_count": len(records),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (partial / "README.md").write_text(
            "# Runtime configuration layer\n\n"
            "This layer was added after the immutable copied input/output archive. "
            "`00_common` and `01_launchers` are subject-agnostic; `02_subject_examples` contains only example-specific provenance. "
            "Verify every file with `runtime_configuration_manifest.jsonl`.\n",
            encoding="utf-8",
        )
        os.replace(partial, target)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--code-root", required=True)
    args = parser.parse_args()
    print(materialize_runtime_configuration(args.archive_root, args.code_root))


if __name__ == "__main__":
    main()
