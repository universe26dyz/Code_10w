"""Immutable per-run provenance written by the server-only Phase-4R runners."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
from .contracts import BASELINE_POSE_POLICY, PROPOSED_CONTRACT, sha256, write_immutable_json


def _available_observation_hashes(root: str | Path) -> dict[str, str]:
    """Hash the observations that are actually present in a sparse case.

    Profile-v2 manifests may intentionally omit 2CH and/or 4CH training
    stacks.  The strict ``observation_hashes`` contract remains unchanged for
    full-data callers; per-run provenance must instead describe the concrete
    sparse input that was used.
    """
    root = Path(root)
    result = {}
    for stack in ("sax", "2ch", "4ch"):
        path = root / stack / "observations.npz"
        if path.is_file():
            result[stack] = sha256(path)
    if not result:
        raise FileNotFoundError(f"no prepared observations found under {root}")
    return result

def write_run_provenance(output: str | Path, method: str, subject: str, prepared_root: str | Path, decoder: str | Path | None = None, b6_model: str | Path | None = None, nesvor_executable: str | Path | None = None) -> Path:
    root = Path(output); prepared = Path(prepared_root) / subject
    if not prepared.exists(): prepared = Path(prepared_root)
    files = [path for path in sorted(root.rglob("*")) if path.is_file() and path.name != "run_provenance.json"]
    input_hashes = _available_observation_hashes(prepared)
    sparse_manifest = prepared / "sparse_manifest.json"
    payload = {"subject": subject, "method": method, "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()), "input_observation_sha256": input_hashes, "input_observation_stacks": sorted(input_hashes), "sparse_manifest_sha256": sha256(sparse_manifest) if sparse_manifest.is_file() else None, "decoder_sha256": sha256(decoder) if decoder else None, "b6_sha256": sha256(b6_model) if b6_model else None, "nesvor_executable": str(nesvor_executable) if nesvor_executable else None, "resolved_scientific_controls": PROPOSED_CONTRACT if method == "proposed" else BASELINE_POSE_POLICY, "output_sha256": {str(path.relative_to(root)): sha256(path) for path in files}}
    return write_immutable_json(root / "run_provenance.json", payload)

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True); parser.add_argument("--method",choices=("proposed","fit_first"),required=True); parser.add_argument("--subject",required=True); parser.add_argument("--prepared-root",required=True); parser.add_argument("--decoder"); parser.add_argument("--b6-model"); parser.add_argument("--nesvor-executable"); args=parser.parse_args(); print(write_run_provenance(**vars(args)))
if __name__ == "__main__": main()
