from pathlib import Path

import pytest

from tygpt.config import Config, ConfigError, apply_overrides, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_yaml_file(tmp_path):
    cfg = load_config(write(tmp_path, "model:\n  n_layer: 3\ntrain:\n  lr: 3e-4\n"))
    assert cfg.model.n_layer == 3
    assert cfg.train.lr == pytest.approx(3e-4)


def test_empty_file_gives_defaults(tmp_path):
    assert load_config(write(tmp_path, "")) == Config()


def test_override_replaces_file_value(tmp_path):
    path = write(tmp_path, "train:\n  lr: 0.001\n")
    cfg = load_config(path, ["train.lr=5e-4", "model.n_layer=2"])
    assert cfg.train.lr == pytest.approx(5e-4)
    assert cfg.model.n_layer == 2


def test_override_can_set_null(tmp_path):
    path = write(tmp_path, "train:\n  max_hours: 4\n")
    assert load_config(path, ["train.max_hours=null"]).train.max_hours is None


def test_override_value_may_contain_equals_sign():
    out = apply_overrides({}, ["paths.data_dir=/content/a=b"])
    assert out["paths"]["data_dir"] == "/content/a=b"


def test_apply_overrides_does_not_mutate_input():
    raw = {"train": {"lr": 0.1}}
    out = apply_overrides(raw, ["train.lr=0.2"])
    assert raw == {"train": {"lr": 0.1}}
    assert out["train"]["lr"] == 0.2


@pytest.mark.parametrize("bad", ["train.lr", "lr=0.1", "train.lr.x=1", ".lr=1", "train.=1"])
def test_malformed_override_is_rejected(bad):
    with pytest.raises(ConfigError, match="override"):
        apply_overrides({}, [bad])


def test_override_with_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key 'train.lrr'"):
        load_config(write(tmp_path, ""), ["train.lrr=1"])


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write(tmp_path, "model: [unclosed\n"))


def test_top_level_list_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- a\n- b\n"))


def test_at_least_one_config_is_shipped():
    assert list(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_every_shipped_config_loads(path):
    load_config(path)


def test_tiny_cpu_config_values():
    cfg = load_config(CONFIG_DIR / "tiny_cpu.yaml")
    assert (cfg.model.n_layer, cfg.model.n_head, cfg.model.d_model) == (4, 4, 128)
    assert cfg.train.device == "cpu"
