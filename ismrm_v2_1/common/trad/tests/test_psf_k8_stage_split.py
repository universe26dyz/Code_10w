import json
from pathlib import Path

import numpy as np
import pytest

from trad.evaluation.scmr.psf_k8_postprocess_common import verify_stage1
from trad.evaluation.scmr.reference_2d import STACKS, sha256


def _stage1(root: Path, n_samples: int = 8) -> None:
    signals = root / "signals"; signals.mkdir(parents=True)
    outputs = {}
    for stack in STACKS:
        path = signals / f"signal_reprojection_{stack}.npz"
        values = np.ones((10, 2, 2), dtype=np.float32)
        np.savez_compressed(path, observed=values, predicted=values, residual=np.zeros_like(values), masks=np.ones_like(values, dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64))
        outputs[stack] = {"relative_path": str(path.relative_to(root)), "sha256": sha256(path)}
    manifest = {"status": "PASS", "subject_id": "CYJ", "psf": {"enabled": True, "n_samples": n_samples, "seed": 20260921}, "signal_outputs": outputs}
    (root / "stage1_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "TRANSFER_SHA256SUMS.txt").write_text("".join(f"{value['sha256']}  {value['relative_path']}\n" for value in outputs.values()), encoding="utf-8")


def test_stage1_provenance_validation_accepts_k8_and_rejects_other_k(tmp_path):
    valid = tmp_path / "valid"; _stage1(valid)
    assert verify_stage1(valid, "CYJ")["psf"]["n_samples"] == 8
    invalid = tmp_path / "invalid"; _stage1(invalid, n_samples=7)
    with pytest.raises(ValueError, match="K=8"):
        verify_stage1(invalid, "CYJ")


def test_stage1_provenance_validation_rejects_tampered_signal(tmp_path):
    root = tmp_path / "stage1"; _stage1(root)
    with (root / "signals" / "signal_reprojection_sax.npz").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="hash"):
        verify_stage1(root, "CYJ")


def test_stage_boundaries_keep_matlab_and_checkpoint_out_of_the_wrong_stage():
    root = Path(__file__).resolve().parents[1] / "evaluation" / "scmr"
    stage1 = (root / "run_psf_k8_gpu_stage.py").read_text(encoding="utf-8").lower()
    stage2 = (root / "run_psf_k8_postprocess.py").read_text(encoding="utf-8").lower()

    assert "matlab" not in stage1
    assert "mask_bundle" not in stage1
    assert "checkpoint" not in stage2
    assert "tinycudann" not in stage2
