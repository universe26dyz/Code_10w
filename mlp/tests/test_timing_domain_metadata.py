import numpy as np

from modules.module_05_signal_decoder.synthetic_dataset import timing_domain_metadata


def test_timing_domain_metadata_keeps_train_and_complete_pool_ranges_separate():
    timing = np.asarray([[10.0] * 9, [20.0] * 9, [30.0] * 9])
    subject = np.asarray(["S001", "S002", "S003"])
    result = timing_domain_metadata(timing, subject, {"train": ["S001"], "valid": ["S002"], "test": ["S003"]})
    assert result["train_timing9_min_ms"] == [10.0] * 9
    assert result["train_timing9_max_ms"] == [10.0] * 9
    assert result["pool_timing9_max_ms"] == [30.0] * 9
    assert result["test_timing_relation_to_train"] == "extrapolation_from_train_domain"
