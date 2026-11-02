from pathlib import Path


def test_checkpoint_guide_requires_explicit_approved_checkpoint_and_controlled_output_root():
    text = (Path(__file__).resolve().parents[2] / "MLP_SURROGATE_CHECKPOINT_INTEGRATION.md").read_text(encoding="utf-8")
    assert "--mlp-checkpoint" in text
    assert "approved_by_manual_review" in text
    assert "mlp_surrogate_v1/CYJ_B6" in text
    assert "不修改" in text
