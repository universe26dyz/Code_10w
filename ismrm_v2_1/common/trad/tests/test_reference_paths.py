from pathlib import Path


def test_reference_path_config_warns_and_skips_when_bundles_are_absent(tmp_path):
    from evaluation.scmr.reference_paths import load_reference_paths

    config = tmp_path / "reference_paths.yaml"
    config.write_text(
        "reference_data:\n"
        "  native_reference_root: /missing/native_reference\n"
        "  myocardium_mask_root: /missing/myocardium_masks\n",
        encoding="utf-8",
    )

    result = load_reference_paths(config)

    assert result.native_reference_root is None
    assert result.myocardium_mask_root is None
    assert result.skip_metrics
    assert any("native reference" in warning.lower() for warning in result.warnings)
    assert any("myocardium" in warning.lower() for warning in result.warnings)


def test_reference_path_config_accepts_complete_bundles(tmp_path):
    from evaluation.scmr.reference_paths import load_reference_paths

    native = tmp_path / "native"; native.mkdir()
    masks = tmp_path / "masks"; masks.mkdir()
    (native / "native_reference_manifest.json").write_text("{}", encoding="utf-8")
    (masks / "manifest.json").write_text('{"status":"PASS"}', encoding="utf-8")
    (masks / "sax_myocardium_masks.npz").write_bytes(b"test")
    config = tmp_path / "reference_paths.yaml"
    config.write_text(
        f"reference_data:\n  native_reference_root: {native}\n  myocardium_mask_root: {masks}\n",
        encoding="utf-8",
    )

    result = load_reference_paths(config)

    assert result.native_reference_root == native
    assert result.myocardium_mask_root == masks
    assert not result.skip_metrics
    assert result.warnings == ()
