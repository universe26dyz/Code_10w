from pathlib import Path

import numpy as np


LEGACY_ROOT = Path("/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first")
EXPECTED_SHA256 = {
    "lipari.txt": "59a26ca79c66da1674f21e4ef83dd6a428c7f1e768327fc4b09e3242b7b81829",
    "navia.txt": "79c6235b421a120e24694dfac67bbbcbc21a3faed0a8a7c72ad8d3c7c41acbfb",
}


def test_vendored_scmr_luts_are_exact_legacy_files_and_construct_identical_colormaps():
    from trad.evaluation.scmr.quality_control import _sha256, load_legacy_scmr_colorbars

    legacy = load_legacy_scmr_colorbars(LEGACY_ROOT)
    vendored = load_legacy_scmr_colorbars()
    asset_root = Path(__file__).resolve().parents[1] / "evaluation" / "scmr" / "assets" / "colormaps"
    samples = np.linspace(0.0, 1.0, 257)
    for parameter, filename in (("T1", "lipari.txt"), ("T2", "navia.txt")):
        legacy_path = LEGACY_ROOT / "python" / "visualization" / "assets" / "colormaps" / filename
        vendored_path = asset_root / filename
        assert _sha256(legacy_path) == EXPECTED_SHA256[filename]
        assert _sha256(vendored_path) == EXPECTED_SHA256[filename]
        assert legacy_path.read_bytes() == vendored_path.read_bytes()
        assert np.array_equal(np.loadtxt(legacy_path), np.loadtxt(vendored_path))
        assert legacy.mapping[parameter].N == vendored.mapping[parameter].N == 256
        assert np.array_equal(legacy.mapping[parameter](samples), vendored.mapping[parameter](samples))
        assert np.array_equal(vendored.mapping[parameter](np.ma.masked), np.array([0.0, 0.0, 0.0, 1.0]))
        assert np.array_equal(vendored.mapping[parameter](-0.1), np.array([0.0, 0.0, 0.0, 1.0]))
    assert vendored.metadata["asset_source_type"] == "vendored_exact_legacy_asset"
    assert legacy.metadata["asset_source_type"] == "external_legacy_override"
