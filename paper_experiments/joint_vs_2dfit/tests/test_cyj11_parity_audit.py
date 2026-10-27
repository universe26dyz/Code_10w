from __future__ import annotations

import json

import numpy as np


def test_current_parity_audit_records_per_plane_support_and_mean(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_parity_audit import write_current_parity_audit

    native = {"group_idx": np.array([4]), "affine_lps_rc": np.eye(4)[None],
              "t1_ms": np.array([[[1000., 1100.], [1200., 1300.]]]),
              "t2_ms": np.array([[[40., 45.], [50., 55.]]])}
    masks = {"myocardium_full": np.ones((1, 2, 2), bool),
             "myocardium_core_legacy": np.array([[[1, 1], [0, 0]]], bool),
             "myocardium_core_1px": np.array([[[0, 1], [0, 0]]], bool)}
    planes = {4: {"roi_slice_index": 8}}
    endpoint_planes = {
        (4, "t1"): {"proposed": native["t1_ms"][0] + 5, "fit_first": native["t1_ms"][0] - 10,
                    "proposed_valid": np.ones((2, 2), bool), "fit_first_valid": np.array([[1, 1], [0, 1]], bool)},
        (4, "t2"): {"proposed": native["t2_ms"][0] + 1, "fit_first": native["t2_ms"][0] - 2,
                    "proposed_valid": np.ones((2, 2), bool), "fit_first_valid": np.ones((2, 2), bool)},
    }

    output = write_current_parity_audit(native=native, masks=masks, endpoint_planes=endpoint_planes,
                                        subject="CYJ", case="L100_fold0", plane_metadata=planes,
                                        output=tmp_path / "current.json")
    audit = json.loads(output.read_text())
    t1 = audit["planes"][0]["parameters"]["t1"]
    assert audit["schema"] == "cyj11_parity_current_native/v1"
    assert audit["planes"][0]["native_group_idx"] == 4
    assert audit["planes"][0]["manual_roi_slice_index"] == 8
    assert t1["common_support_voxels"] == 3
    assert t1["core_legacy"]["support_voxels"] == 2
    assert t1["core_legacy"]["reference_mean"] == 1050.0
