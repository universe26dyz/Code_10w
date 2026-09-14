import importlib.util
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]
TRAD_ROOT = CODE_ROOT / "trad"


def _load_script():
    for path in (CODE_ROOT, TRAD_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    script = TRAD_ROOT / "scripts" / "prepare_all_observations.py"
    spec = importlib.util.spec_from_file_location("prepare_all_observations_deployment", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_all_writes_one_explicit_output_per_manifest_entry(tmp_path, monkeypatch):
    module = _load_script()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": "deployment-v1", "subjects": [
        {"subject_id": "S01", "stacks": [{"stack": "sax", "dicom_dir": "/raw/S01/sax"}]}
    ]}), encoding="utf-8")
    preprocessed_root, prepared_root = tmp_path / "pre", tmp_path / "prepared"
    mat = preprocessed_root / "S01" / "sax" / "preprocessed.mat"
    mat.parent.mkdir(parents=True); mat.touch()

    def fake_prepare(preprocessed_mat, dicom_dir, stack, output_dir, max_groups=None):
        output = Path(output_dir); output.mkdir(parents=True)
        for name in ("observations.npz", "manifest.json", "timing.npy", "qc_summary.json"):
            (output / name).write_text(name, encoding="utf-8")
        return {"qc_summary": {"group_count": 1, "weights_per_group": [10], "timing_shape": [1, 9], "geometry_shape": [1, 4, 4]}}

    monkeypatch.setattr(module, "prepare_observations", fake_prepare)
    result = module.prepare_all_observations(manifest, preprocessed_root, prepared_root)
    assert result["prepared_count"] == 1
    assert (prepared_root / "S01" / "sax" / "observations.npz").is_file()
    batch = json.loads((prepared_root / "PREPARED_BATCH_QC.json").read_text(encoding="utf-8"))
    assert batch["entries"][0]["subject_id"] == "S01"
