"""Immutable, operator-selected Phase-4R profile and proposed-method contract."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from .contracts import APPROVED_DECODER_SHA256, B6_CONFIG_SOURCE_SHA256, sha256, write_immutable_json


SCHEMA = "phase4r_final_method_selection/v1"
PROPOSED_METHODS = {
    "Q002S640": {"experiment_id": "Q002S640_B6_roi_same_anchor_mse", "training_psf_samples": 8},
    "Q005S640_K4": {"experiment_id": "Q005S640_K4_B6_roi_same_anchor_mse", "training_psf_samples": 4},
}


def _record(path: str | Path) -> dict[str, str]:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(value)
    return {"path": str(value.resolve()), "sha256": sha256(value)}


def validate_final_method_selection(manifest: str | Path) -> dict[str, Any]:
    path = Path(manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("immutable") is not True:
        raise ValueError("invalid final method selection schema or immutability")
    method = payload.get("selected_proposed_method")
    if method not in PROPOSED_METHODS or payload.get("training_psf_samples") != PROPOSED_METHODS[method]["training_psf_samples"]:
        raise ValueError("selected proposed method and training PSF contract disagree")
    profile = payload.get("selected_profile", {})
    if not isinstance(profile, dict) or not isinstance(profile.get("profile_id"), str) or not profile.get("profile_id"):
        raise ValueError("final method selection requires an explicit selected profile")
    profile_path = Path(str(profile.get("profile_path", "")))
    if not profile_path.is_file() or sha256(profile_path) != profile.get("profile_sha256"):
        raise ValueError("final method selection selected profile SHA256 mismatch")
    policy = payload.get("profile_replication_policy")
    if not isinstance(policy, dict) or policy.get("mode") not in {"one_preregistered_fold", "all_profile_folds"}:
        raise ValueError("final method selection requires explicit profile fold replication policy")
    if policy["mode"] == "one_preregistered_fold" and not isinstance(policy.get("fold_id"), int):
        raise ValueError("one_preregistered_fold requires an explicit fold_id")
    if payload.get("frozen_dependencies") != {"b6_sha256": B6_CONFIG_SOURCE_SHA256, "decoder_sha256": APPROVED_DECODER_SHA256, "protocol_sha256": payload.get("frozen_dependencies", {}).get("protocol_sha256")} or not isinstance(payload["frozen_dependencies"].get("protocol_sha256"), str):
        raise ValueError("final method selection frozen dependency contract is invalid")
    for category in ("experiment_1_1_evidence", "experiment_1_2_evidence", "metric_figure_summary_evidence"):
        evidence = payload.get(category)
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError(f"final method selection lacks {category}")
        for item in evidence.values():
            candidate = Path(str(item.get("path", ""))) if isinstance(item, dict) else Path("")
            if not candidate.is_file() or sha256(candidate) != item.get("sha256"):
                raise ValueError(f"final method selection {category} SHA256 mismatch")
    return payload


def create_final_method_selection(*, output: str | Path, selected_profile: dict[str, Any], selected_proposed_method: str, profile_replication_policy: dict[str, Any], experiment_1_1_evidence: dict[str, str | Path], experiment_1_2_evidence: dict[str, str | Path], metric_figure_summary_evidence: dict[str, str | Path], protocol: str | Path) -> Path:
    """Write an immutable explicit decision; never calculate or infer a winner."""
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    if selected_proposed_method not in PROPOSED_METHODS:
        raise ValueError("selected proposed method is not a formal Q002S640/Q005S640_K4 route")
    if not isinstance(selected_profile, dict) or not selected_profile.get("profile_id"):
        raise ValueError("selected profile must be supplied explicitly")
    profile_path = Path(str(selected_profile.get("profile_path", "")))
    if not profile_path.is_file() or sha256(profile_path) != selected_profile.get("profile_sha256"):
        raise ValueError("selected profile must include its exact frozen JSON path and SHA256")
    if not all((experiment_1_1_evidence, experiment_1_2_evidence, metric_figure_summary_evidence)):
        raise ValueError("final selection requires explicit Experiment 1.1, Experiment 1.2, and metric/figure evidence")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA, "immutable": True, "selected_profile": selected_profile,
        "selected_proposed_method": selected_proposed_method, "experiment_id": PROPOSED_METHODS[selected_proposed_method]["experiment_id"], "training_psf_samples": PROPOSED_METHODS[selected_proposed_method]["training_psf_samples"],
        "profile_replication_policy": profile_replication_policy,
        "experiment_1_1_evidence": {name: _record(value) for name, value in experiment_1_1_evidence.items()},
        "experiment_1_2_evidence": {name: _record(value) for name, value in experiment_1_2_evidence.items()},
        "metric_figure_summary_evidence": {name: _record(value) for name, value in metric_figure_summary_evidence.items()},
        "frozen_dependencies": {"b6_sha256": B6_CONFIG_SOURCE_SHA256, "decoder_sha256": APPROVED_DECODER_SHA256, "protocol_sha256": sha256(protocol)},
        "selection_code_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write_immutable_json(target, payload)
    validate_final_method_selection(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); payload = json.loads(Path(args.selection_json).read_text(encoding="utf-8")); print(create_final_method_selection(output=args.output, **payload))


if __name__ == "__main__":
    main()
