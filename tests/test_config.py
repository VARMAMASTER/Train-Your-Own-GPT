import pytest

from tygpt.config import Config, ConfigError, ModelConfig, config_from_dict, config_to_dict


def test_empty_dict_gives_defaults():
    assert config_from_dict({}) == Config()


def test_none_gives_defaults():
    assert config_from_dict(None) == Config()


def test_values_are_read_into_sections():
    cfg = config_from_dict(
        {"model": {"n_layer": 6, "n_head": 6, "d_model": 384}, "train": {"lr": 0.001}}
    )
    assert cfg.model.n_layer == 6
    assert cfg.model.d_model == 384
    assert cfg.train.lr == 0.001
    assert cfg.model.block_size == ModelConfig().block_size


def test_scientific_notation_string_becomes_float():
    # YAML 1.1 reads "3e-4" (no decimal point) as a string, not a float.
    cfg = config_from_dict({"train": {"lr": "3e-4"}})
    assert cfg.train.lr == pytest.approx(3e-4)


def test_int_given_for_float_field_is_converted():
    cfg = config_from_dict({"train": {"weight_decay": 0}})
    assert isinstance(cfg.train.weight_decay, float)


def test_optional_field_accepts_null_and_number():
    assert config_from_dict({"train": {"max_hours": None}}).train.max_hours is None
    assert config_from_dict({"train": {"max_hours": 6}}).train.max_hours == 6.0


def test_unknown_section_is_rejected():
    with pytest.raises(ConfigError, match="unknown section 'modle'"):
        config_from_dict({"modle": {}})


def test_unknown_key_is_rejected():
    with pytest.raises(ConfigError, match="unknown key 'train.learning_rate'"):
        config_from_dict({"train": {"learning_rate": 1e-3}})


def test_section_must_be_a_mapping():
    with pytest.raises(ConfigError, match="section 'train' must be a mapping"):
        config_from_dict({"train": 5})


def test_wrong_type_is_rejected():
    with pytest.raises(ConfigError, match="model.n_layer"):
        config_from_dict({"model": {"n_layer": "six"}})


def test_bool_is_not_accepted_as_int():
    with pytest.raises(ConfigError, match="model.n_layer"):
        config_from_dict({"model": {"n_layer": True}})


def test_d_model_must_divide_by_heads():
    with pytest.raises(ConfigError, match="divisible"):
        config_from_dict({"model": {"d_model": 130, "n_head": 4}})


def test_model_vocab_must_fit_tokenizer_vocab():
    with pytest.raises(ConfigError, match="model.vocab_size"):
        config_from_dict({"tokenizer": {"vocab_size": 16384}, "model": {"vocab_size": 8192}})


@pytest.mark.parametrize(
    "section,key,value,message",
    [
        ("tokenizer", "backend", "sentencepiece", "tokenizer.backend"),
        ("train", "device", "tpu", "train.device"),
        ("train", "dtype", "int8", "train.dtype"),
        ("data", "source_type", "csv", "data.source_type"),
        ("data", "val_fraction", 0.0, "data.val_fraction"),
        ("train", "lr", 0.0, "train.lr"),
        ("model", "n_layer", 0, "model.n_layer"),
        ("model", "dropout", 1.0, "model.dropout"),
        ("train", "max_hours", 0, "train.max_hours"),
        ("tokenizer", "vocab_size", 70000, "uint16"),
    ],
)
def test_invalid_values_are_rejected(section, key, value, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict({section: {key: value}})


def test_all_problems_reported_at_once():
    with pytest.raises(ConfigError) as exc:
        config_from_dict({"train": {"lr": 0.0, "device": "tpu"}})
    assert "train.lr" in str(exc.value)
    assert "train.device" in str(exc.value)


def test_to_dict_round_trips():
    cfg = config_from_dict({"model": {"n_layer": 2}, "train": {"max_hours": 3}})
    assert config_from_dict(config_to_dict(cfg)) == cfg
