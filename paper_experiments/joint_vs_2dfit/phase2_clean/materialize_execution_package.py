"""Create a visible, runnable code-and-runtime layer beside frozen evidence."""
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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_commit(code: Path) -> str:
    status = subprocess.run(["git", "-C", str(code), "status", "--porcelain"], check=True, text=True, capture_output=True).stdout
    tracked = [line for line in status.splitlines() if not line.startswith("??")]
    if tracked:
        raise ValueError("refusing execution package from tracked modified code: " + "; ".join(tracked))
    return subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()


def _copy_file(source: Path, destination: Path, root: Path, label: str, records: list[dict[str, str]]) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required execution-package file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    records.append({"label": label, "source": str(source), "archive": str(destination.relative_to(root)), "sha256": _sha256(destination)})


def _copy_tree(source: Path, destination: Path, root: Path, label: str, records: list[dict[str, str]]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"required execution-package directory is missing: {source}")
    shutil.copytree(source, destination)
    for copied in sorted(path for path in destination.rglob("*") if path.is_file()):
        records.append({"label": label, "source": str(source / copied.relative_to(destination)), "archive": str(copied.relative_to(root)), "sha256": _sha256(copied)})


def _copy_tracked_code(code: Path, destination: Path, root: Path, records: list[dict[str, str]]) -> None:
    listed = subprocess.run(["git", "-C", str(code), "ls-files", "-z"], check=True, capture_output=True).stdout.decode().split("\0")
    for relative_text in (item for item in listed if item):
        relative = Path(relative_text)
        _copy_file(code / relative, destination / relative, root, "tracked_project_code", records)


def _write_code_map(root: Path) -> None:
    guides = {
        "01_server_dicom_preprocessing.md": [
            "# 01 Server DICOM preprocessing", "",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/server_preprocess_and_prepare.sh`",
            "- `code/trad/scripts/local_preprocess_all.m` (historical name; executed on the server by the launcher)",
            "- `code/trad/scripts/local_preprocess_one.m`",
            "- `code/trad/modules/module_01_preprocess_matlab/preprocessing_v1/`",
        ],
        "02_server_observation_bridge.md": [
            "# 02 Server observation bridge", "",
            "- `code/trad/scripts/validate_transferred_preprocessed.py`", "- `code/trad/scripts/prepare_all_observations.py`",
            "- `code/trad/modules/module_02_data_bridge/`",
        ],
        "03_server_sparse_profiles.md": [
            "# 03 Server sparse-profile construction", "",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh`",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`",
            "- `code/paper_experiments/joint_vs_2dfit/configs/sparse_profiles/`",
        ],
        "04_scmr_mapping_and_fit_first.md": [
            "# 04 SCMR-v5 mapping and 2D-fit-first", "",
            "- `code/trad/evaluation/scmr/matlab/`", "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/scmr_v5_native_cache.py`",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh`",
        ],
        "05_proposed_reconstruction.md": [
            "# 05 Proposed reconstruction", "",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/proposed.py`", "- `code/quality_experiments/Q002_B6_roi_same_anchor_mse/`",
            "- `code/mlp/`, `code/reconstruction_core/`, and `code/trad/modules/`",
        ],
        "06_final_pose_metrics_visualization.md": [
            "# 06 Final pose, metrics, and visualization", "",
            "- `code/paper_experiments/joint_vs_2dfit/aha/`", "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh`",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh`",
            "- `code/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh`",
        ],
    }
    target = root / "00_pipeline_code_map"
    target.mkdir()
    for name, lines in guides.items():
        (target / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_execution_package(
    *, archive_root: str | Path, code_root: str | Path, decoder: str | Path, b6_model: str | Path,
    nesvor_package_root: str | Path, nesvor_executable: str | Path, cr_dreme_pip_freeze: str | Path,
    nesvor_pip_freeze: str | Path, local_matlab_report: str | Path, server_matlab_report: str | Path,
) -> Path:
    archive, code = Path(archive_root).resolve(), Path(code_root).resolve()
    manifest = _load(archive / "99_integrity" / "archive_manifest.json")
    if manifest.get("schema_version") != "scmr_v5_multimap_reproducibility_archive/v1":
        raise ValueError("not a completed SCMR-v5 reproducibility archive")
    commit = _code_commit(code)
    target = archive / "01_execution_code"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite visible execution code: {target}")
    partial = Path(tempfile.mkdtemp(prefix=".01_execution_code.partial-", dir=archive))
    try:
        records: list[dict[str, str]] = []
        _copy_tracked_code(code, partial / "code", partial, records)
        _write_code_map(partial)
        _copy_file(Path(decoder), partial / "01_model_assets" / "approved_decoder.pth", partial, "model_asset", records)
        _copy_file(Path(b6_model), partial / "01_model_assets" / "b6_model.pt", partial, "model_asset", records)
        _copy_tree(Path(nesvor_package_root), partial / "02_external_runtime" / "nesvor_package", partial, "external_runtime", records)
        _copy_file(Path(nesvor_executable), partial / "02_external_runtime" / "bin" / "nesvor", partial, "external_runtime", records)
        _copy_file(Path(cr_dreme_pip_freeze), partial / "03_environment_records" / "cr_dreme_pip_freeze.txt", partial, "environment", records)
        _copy_file(Path(nesvor_pip_freeze), partial / "03_environment_records" / "nesvor_pip_freeze.txt", partial, "environment", records)
        _copy_file(Path(local_matlab_report), partial / "03_environment_records" / "local_matlab_version.txt", partial, "environment", records)
        _copy_file(Path(server_matlab_report), partial / "03_environment_records" / "server_matlab_version.txt", partial, "environment", records)
        integrity = partial / "99_integrity"
        integrity.mkdir()
        (integrity / "execution_manifest.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8"
        )
        (integrity / "execution_metadata.json").write_text(json.dumps({
            "schema_version": "scmr_v5_execution_package/v1", "code_root": str(code), "code_commit": commit,
            "file_count": len(records), "source_archive_code_snapshot_commit": manifest.get("code_snapshot", {}).get("commit"),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (partial / "README.md").write_text(
            "# Visible execution package\n\n"
            "`code/` is an uncompressed copy of every tracked project file at the recorded commit. "
            "`00_pipeline_code_map/` maps execution stages to their working source files. "
            "Model assets, the actual external NeSVoR package/CLI, and environment records are separately hashed in `99_integrity/`.\n",
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
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--b6-model", required=True)
    parser.add_argument("--nesvor-package-root", required=True)
    parser.add_argument("--nesvor-executable", required=True)
    parser.add_argument("--cr-dreme-pip-freeze", required=True)
    parser.add_argument("--nesvor-pip-freeze", required=True)
    parser.add_argument("--local-matlab-report", required=True)
    parser.add_argument("--server-matlab-report", required=True)
    print(materialize_execution_package(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
