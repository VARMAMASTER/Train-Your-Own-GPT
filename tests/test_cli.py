from pathlib import Path

import pytest
import yaml

from tygpt.cli import main

TINY = Path(__file__).resolve().parents[1] / "configs" / "tiny_cpu.yaml"


def test_config_command_prints_resolved_config(capsys):
    assert main(["config", str(TINY), "train.lr=5e-4"]) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["train"]["lr"] == pytest.approx(5e-4)
    assert printed["model"]["n_layer"] == 4


def test_config_error_returns_1_with_message(capsys):
    assert main(["config", str(TINY), "train.nope=1"]) == 1
    assert "error: unknown key 'train.nope'" in capsys.readouterr().err


def test_missing_config_file_returns_1(capsys, tmp_path):
    assert main(["config", str(tmp_path / "missing.yaml")]) == 1
    assert "not found" in capsys.readouterr().err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tygpt 0.1.0" in capsys.readouterr().out


def test_no_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
