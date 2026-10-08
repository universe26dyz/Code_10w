from pathlib import Path


def test_q002_server_scripts_are_isolated_and_protected():
    root = Path(__file__).resolve().parents[2] / "server_commands"
    run, evaluate = (root / "run_Q002_CYJ.sh").read_text(), (root / "evaluate_Q002_CYJ.sh").read_text()
    for text in (run, evaluate):
        assert "conda activate cr_dreme" in text and "git status --porcelain" in text
        assert "rm -rf" not in text and "Q001" not in text and "full_fov" not in text.lower()
    assert "torch.cuda.is_available" in run and "sha256sum" in run
    assert "Refusing non-empty output" in run and "Refusing existing log" in run
    assert "Refusing non-empty evaluation" in evaluate
