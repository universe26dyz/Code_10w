import ast
from pathlib import Path


def test_raw_export_precedes_psf128_export_in_q002_launcher():
    source = Path(__file__).resolve().parents[1] / "run_q002_reconstruction.py"
    tree = ast.parse(source.read_text())
    calls = [node.func.id if isinstance(node.func, ast.Name) else node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert calls.index("export_quantitative_outputs") < calls.index("export_quantitative_outputs_psf128")


def test_signal_evaluator_uses_matched_b6_baseline_index_not_candidate_row():
    source = (Path(__file__).resolve().parents[1] / "evaluate_q002.py").read_text()
    assert '("FrozenMLP_B6", b6["predicted"][baseline_index])' in source
    assert "load_verified_reference" in source and "signal_agreement_metrics" in source


def test_observed_fingerprint_cosine_uses_epsilon_and_reports_zero_norms():
    import numpy as np
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import fingerprint_cosine_summary

    predicted = np.ones((10, 1, 2), np.float32); observed = predicted.copy(); observed[:, 0, 1] = 0
    result = fingerprint_cosine_summary(predicted, observed, np.ones((1, 2), bool))
    assert result["mean_cosine"] == 0.5 and result["support_N"] == 2 and result["near_zero_norm_count"] == 1
