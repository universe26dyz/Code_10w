"""Actual-file real-only Phase-4R readiness gate; it intentionally has no XCAT dependency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contracts import B6_CONFIG_SOURCE_SHA256, ELIGIBLE_REAL_SUBJECTS, sha256, validate_proposed_subject
from .historical_nesvor_runtime import verify_runtime
from .preflight_actual import verify_approved_decoder


def _verify_observation(path: Path, subject: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared observation: {path}")
    with np.load(path, allow_pickle=False) as data:
        for field in ("images", "masks", "group_idx", "weight_idx", "tr_ms", "vps"):
            if field not in data.files: raise ValueError(f"{path}: missing {field}")
        groups = np.unique(data["group_idx"])
        for group in groups:
            weights = np.sort(data["weight_idx"][data["group_idx"] == group])
            if not np.array_equal(weights, np.arange(10)): raise ValueError(f"{path}: group {int(group)} does not contain weights 0..9")
        tr, vps = float(data["tr_ms"][0]), int(data["vps"][0])
        if not np.allclose(data["tr_ms"], tr) or not np.all(data["vps"] == vps): raise ValueError(f"{path}: mixed TR/VPS")
        validate_proposed_subject(subject, tr, vps)
    return {"path": str(path.resolve()), "sha256": sha256(path), "group_count": int(len(groups)), "tr_ms": tr, "vps": vps, "weight_integrity": True}


def verify_real(
    manifest: str | Path, data_root: str | Path, decoder: str | Path, b6_model: str | Path,
    reference_manifest: str | Path, reference_root: str | Path, approved_decoder_audit: str | Path,
    nesvor_executable: str | Path, nesvor_python: str | Path, nesvor_source: str | Path,
) -> dict[str, object]:
    """Verify only the formal real cohort; no synthetic/XCAT path is accepted or read."""
    expected = json.loads(Path(manifest).read_text())
    references = json.loads(Path(reference_manifest).read_text())
    data_root, reference_root = Path(data_root), Path(reference_root)
    actual: dict[str, dict[str, object]] = {}
    for subject in ELIGIBLE_REAL_SUBJECTS:
        actual[subject] = {}
        for stack in ("sax", "2ch", "4ch"):
            record = _verify_observation(data_root / subject / stack / "observations.npz", subject)
            if record["sha256"] != expected["real_observation_sha256"][subject][stack]["sha256"]:
                raise ValueError(f"{subject}/{stack} SHA mismatch")
            actual[subject][stack] = record
        reference = references["subjects"].get(subject)
        if not reference: raise ValueError(f"missing reference manifest subject {subject}")
        if reference.get("sax_observation_sha256") != actual[subject]["sax"]["sha256"]: raise ValueError(f"{subject} SAX reference source SHA mismatch")
        for key in ("native_reference", "myocardium_bundle"):
            entry = reference[key]; path = Path(entry.get("relative_path", entry.get("path")))
            path = path if path.is_absolute() else reference_root / path
            if not path.is_file() or sha256(path) != entry["sha256"]: raise ValueError(f"{subject} {key} missing or SHA mismatch")
    approved = verify_approved_decoder(decoder, approved_decoder_audit, reference_root)
    if sha256(b6_model) != B6_CONFIG_SOURCE_SHA256: raise ValueError("B6 source SHA mismatch")
    runtime = verify_runtime(nesvor_executable, nesvor_python, nesvor_source)
    return {"formal_real_reconstruction_ready": True, "actual_observation_hash_gate": True, "references_ready": True, "approved_decoder_gate": True, "b6_config_source_gate": True, "actual_observations": actual, "approved_decoder": approved, "b6_source_model": {"path": str(Path(b6_model).resolve()), "sha256": sha256(b6_model)}, **runtime, "scope": "real_only; synthetic/XCAT deliberately not required"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "reference-manifest", "reference-root", "data-root", "decoder", "approved-decoder-audit", "b6-model", "nesvor-executable", "nesvor-python", "nesvor-source", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); output = Path(args.output)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    values = vars(args); result = verify_real(**{key.replace("-", "_"): values[key.replace("-", "_")] for key in values if key != "output"})
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(output)


if __name__ == "__main__":
    main()
