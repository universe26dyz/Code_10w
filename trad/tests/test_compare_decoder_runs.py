import json

from trad.evaluation.scmr.compare_decoder_runs import compare_decoder_runs


def test_compare_decoder_runs_writes_matching_numeric_metric_deltas(tmp_path):
    trad_metrics = tmp_path / "trad_metrics.json"
    mlp_metrics = tmp_path / "mlp_metrics.json"
    trad_timing = tmp_path / "trad_timing.json"
    mlp_timing = tmp_path / "mlp_timing.json"
    trad_metrics.write_text(json.dumps({"global_common_support": {"T1": {"rmse": 10.0}}, "sax_myocardium": {"T2": {"core": {"mae": 2.0}}}}))
    mlp_metrics.write_text(json.dumps({"global_common_support": {"T1": {"rmse": 8.0}}, "sax_myocardium": {"T2": {"core": {"mae": 3.0}}}}))
    trad_timing.write_text(json.dumps({"decoder_type": "Bloch", "run_level_ms": {"total_runtime": 100.0}}))
    mlp_timing.write_text(json.dumps({"decoder_type": "FrozenMLP", "run_level_ms": {"total_runtime": 80.0}}))

    report = compare_decoder_runs(trad_metrics, mlp_metrics, tmp_path / "comparison.json", trad_timing_summary=trad_timing, mlp_timing_summary=mlp_timing)

    assert report["metric_delta_mlp_minus_trad"]["global_common_support.T1.rmse"] == -2.0
    assert report["metric_delta_mlp_minus_trad"]["sax_myocardium.T2.core.mae"] == 1.0
    assert report["timing"]["trad"]["decoder_type"] == "Bloch"
