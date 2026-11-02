from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = [
    "01_RAW_DICOM", "02_PREPROCESS_MATLAB", "03_PREPARE_OBSERVATIONS",
    "04_SPARSE_PROFILES", "05_SCMR_V5_FIT_FIRST", "06_PROPOSED",
    "07_FINAL_POSE", "08_EVALUATION_AND_VISUALIZATION",
    "09_INTEGRITY_AND_REPORTING",
]

def test_ordered_modules_have_bilingual_manuals():
    for module in MODULES:
        path = ROOT / "pipeline" / module
        assert path.is_dir()
        assert (path / "README_zh-CN.md").is_file()
        assert (path / "README_en.md").is_file()

def test_integrity_script_passes():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/verify_project_integrity.py"), str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr + result.stdout

def test_dry_run_resolves_project_root():
    result = subprocess.run([str(ROOT / "scripts/run_pipeline.sh")], env={"PATH": "/usr/bin:/bin", "SUBJECT": "HHZ", "DRY_RUN": "1"}, capture_output=True, text=True)
    assert result.returncode == 0
    assert "DRY-RUN" in result.stdout

