"""Validate the isolated historical NeSVoR 0.6.0rc1 runtime without reconstructing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, Union

HISTORICAL_VERSION = "0.6.0rc1"
REQUIRED_FLAGS = (
    "--n-inference-samples", "--pixel-variance-weight", "--weight-pixel-var-smooth",
    "--scanner-space", "--no-slice-scale", "--single-precision", "--n-samples",
    "--batch-size", "--n-iter", "--output-psf-factor",
)
REQUIRED_SOURCE_FILES = (
    "cli/parsers.py", "cli/commands.py", "inr/models.py", "inr/train.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_fingerprint(source_root: Path) -> Dict[str, object]:
    files = sorted(path for path in source_root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(_sha256(path).encode("ascii") + b"\n")
    required = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError("historical NeSVoR source file missing: {}".format(path))
        required[relative] = _sha256(path)
    return {"tree_sha256": digest.hexdigest(), "file_count": len(files), "required_file_sha256": required}


def _git_provenance(source_root: Path) -> Dict[str, object]:
    root = source_root.parent
    inside = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"], text=True, capture_output=True)
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"git_backed": False}
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], text=True, capture_output=True, check=True).stdout.strip())
    return {"git_backed": True, "git_commit": commit, "git_dirty": dirty}


def verify_runtime(executable: Union[str, Path], runtime_python: Union[str, Path], source_root: Union[str, Path]) -> Dict[str, object]:
    executable_path, python_path, source = Path(executable), Path(runtime_python), Path(source_root).resolve()
    if not executable_path.is_file() or not os.access(str(executable_path), os.X_OK):
        raise FileNotFoundError("historical NeSVoR executable is missing or not executable: {}".format(executable_path))
    if not python_path.is_file() or not os.access(str(python_path), os.X_OK):
        raise FileNotFoundError("historical NeSVoR python is missing or not executable: {}".format(python_path))
    package = subprocess.run([str(python_path), "-c", "import json, nesvor; from nesvor.version import __version__; print(json.dumps({'package_path': nesvor.__file__, 'version': __version__}))"], text=True, capture_output=True)
    if package.returncode != 0:
        raise RuntimeError("historical NeSVoR python cannot import nesvor: {}".format(package.stderr.strip()))
    resolved = json.loads(package.stdout)
    package_path = Path(resolved["package_path"]).resolve()
    try:
        package_path.relative_to(source)
    except ValueError as exc:
        raise ValueError("historical NeSVoR package source mismatch: {} is not under {}".format(package_path, source)) from exc
    if resolved["version"] != HISTORICAL_VERSION:
        raise ValueError("historical NeSVoR version mismatch: {}".format(resolved["version"]))
    help_result = subprocess.run([str(executable_path), "reconstruct", "--help"], text=True, capture_output=True)
    if help_result.returncode != 0:
        raise RuntimeError("historical NeSVoR help failed: {}".format(help_result.stderr.strip()))
    missing = [flag for flag in REQUIRED_FLAGS if flag not in help_result.stdout + help_result.stderr]
    if missing:
        raise ValueError("historical NeSVoR required flag missing: {}".format(", ".join(missing)))
    return {
        "historical_nesvor_runtime_gate": True,
        "historical_nesvor": {
            "executable": str(executable_path), "python": str(python_path), "version": resolved["version"],
            "package_path": str(package_path), "source_root": str(source), "required_flags_verified": list(REQUIRED_FLAGS),
            **_git_provenance(source), **_source_fingerprint(source),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--python", dest="runtime_python", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = verify_runtime(args.executable, args.runtime_python, args.source_root)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError("refusing overwrite: {}".format(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
