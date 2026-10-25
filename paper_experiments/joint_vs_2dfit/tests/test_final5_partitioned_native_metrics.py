"""Contracts for Final5 all/observed/held-out native reporting partitions."""
from __future__ import annotations


def test_partitioned_roles_preserve_all_observed_and_heldout_sets() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_partitioned_native_metrics import partition_group_roles

    roles = {("sax", 0): "observed", ("2ch", 1): "observed", ("2ch", 2): "heldout"}
    assert partition_group_roles(roles) == {
        "all": {("sax", 0), ("2ch", 1), ("2ch", 2)},
        "observed": {("sax", 0), ("2ch", 1)},
        "heldout": {("2ch", 2)},
    }
