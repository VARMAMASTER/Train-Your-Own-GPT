"""Typed run configuration: one dataclass per section, loaded from plain dicts."""

import copy
import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import yaml

SOURCE_TYPES = ("text_dir", "jsonl", "hf_dataset")
TOKENIZER_BACKENDS = ("scratch_bpe", "tiktoken_gpt2")
DEVICES = ("auto", "cpu", "cuda", "mps")
DTYPES = ("auto", "float32", "bfloat16", "float16")
MAX_UINT16_VOCAB = 65536


class ConfigError(ValueError):
    """Raised when a config file, section, key or value is invalid."""


@dataclass
class DataConfig:
    name: str = "tinystories"
    source_type: str = "hf_dataset"
    source: str = "roneneldan/TinyStories"
    val_fraction: float = 0.01
    seed: int = 1337


@dataclass
class TokenizerConfig:
    backend: str = "scratch_bpe"
    vocab_size: int = 8192


@dataclass
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    block_size: int = 256
    vocab_size: int = 8192
    dropout: float = 0.0
    bias: bool = True


@dataclass
class TrainConfig:
    max_steps: int = 1000
    micro_batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_steps: int = 20
    ckpt_interval: int = 500
    seed: int = 1337
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = False
    max_hours: Optional[float] = None
    max_cost_usd: Optional[float] = None
    cost_per_hour_usd: float = 0.0


@dataclass
class PathsConfig:
    data_dir: str = "data"
    ckpt_dir: str = "checkpoints"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _type_error(where: str, expected: str, value: Any) -> ConfigError:
    return ConfigError(f"{where}: expected {expected}, got {value!r}")


def _coerce(value: Any, tp: Any, where: str) -> Any:
    """Convert a YAML value to the dataclass field type, or raise ConfigError."""
    origin = typing.get_origin(tp)
    if origin is typing.Union or origin is types.UnionType:
        if value is None:
            return None
        inner = [a for a in typing.get_args(tp) if a is not type(None)]
        return _coerce(value, inner[0], where)
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise _type_error(where, "true or false", value)
    if tp is int:
        if isinstance(value, bool):
            raise _type_error(where, "an integer", value)
        if isinstance(value, int):
            return value
        if isinstance(value, (float, str)):
            try:
                as_float = float(value)
            except ValueError:
                raise _type_error(where, "an integer", value) from None
            if as_float.is_integer():
                return int(as_float)
        raise _type_error(where, "an integer", value)
    if tp is float:
        if isinstance(value, bool):
            raise _type_error(where, "a number", value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        raise _type_error(where, "a number", value)
    if tp is str:
        if isinstance(value, str):
            return value
        raise _type_error(where, "a string", value)
    raise ConfigError(f"{where}: unsupported field type {tp!r}")


def config_from_dict(raw: Optional[dict]) -> Config:
    """Build a validated Config from a dict of sections. Missing values use defaults."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping of sections")
    section_types = typing.get_type_hints(Config)
    for name in raw:
        if name not in section_types:
            raise ConfigError(
                f"unknown section '{name}'. Valid sections: {', '.join(section_types)}"
            )
    sections = {}
    for name, section_cls in section_types.items():
        section_raw = raw.get(name)
        if section_raw is None:
            section_raw = {}
        if not isinstance(section_raw, dict):
            raise ConfigError(f"section '{name}' must be a mapping of key: value")
        field_types = typing.get_type_hints(section_cls)
        kwargs = {}
        for key, value in section_raw.items():
            if key not in field_types:
                raise ConfigError(
                    f"unknown key '{name}.{key}'. Valid keys: {', '.join(field_types)}"
                )
            kwargs[key] = _coerce(value, field_types[key], f"{name}.{key}")
        sections[name] = section_cls(**kwargs)
    cfg = Config(**sections)
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    """Plain nested dict, safe to dump as YAML or store in a checkpoint."""
    return dataclasses.asdict(cfg)


def validate(cfg: Config) -> None:
    """Check cross-field rules. Raises ConfigError listing every problem found."""
    errors = []
    m, t, tok, d = cfg.model, cfg.train, cfg.tokenizer, cfg.data

    positive_ints = {
        "model.n_layer": m.n_layer,
        "model.n_head": m.n_head,
        "model.d_model": m.d_model,
        "model.block_size": m.block_size,
        "model.vocab_size": m.vocab_size,
        "tokenizer.vocab_size": tok.vocab_size,
        "train.max_steps": t.max_steps,
        "train.micro_batch_size": t.micro_batch_size,
        "train.grad_accum_steps": t.grad_accum_steps,
        "train.eval_interval": t.eval_interval,
        "train.eval_steps": t.eval_steps,
        "train.ckpt_interval": t.ckpt_interval,
    }
    for where, value in positive_ints.items():
        if value <= 0:
            errors.append(f"{where} must be > 0, got {value}")

    if m.n_head > 0 and m.d_model % m.n_head != 0:
        errors.append(
            f"model.d_model ({m.d_model}) must be divisible by model.n_head ({m.n_head})"
        )
    if not 0.0 <= m.dropout < 1.0:
        errors.append(f"model.dropout must be in [0, 1), got {m.dropout}")
    if tok.vocab_size > MAX_UINT16_VOCAB:
        errors.append(
            f"tokenizer.vocab_size must be <= {MAX_UINT16_VOCAB} because token shards "
            f"are stored as uint16, got {tok.vocab_size}"
        )
    if m.vocab_size < tok.vocab_size:
        errors.append(
            f"model.vocab_size ({m.vocab_size}) must be >= tokenizer.vocab_size ({tok.vocab_size})"
        )
    if not 0.0 < d.val_fraction < 1.0:
        errors.append(f"data.val_fraction must be between 0 and 1, got {d.val_fraction}")
    if t.lr <= 0:
        errors.append(f"train.lr must be > 0, got {t.lr}")
    if not 0.0 < t.min_lr_ratio <= 1.0:
        errors.append(f"train.min_lr_ratio must be in (0, 1], got {t.min_lr_ratio}")
    if t.warmup_steps < 0:
        errors.append(f"train.warmup_steps must be >= 0, got {t.warmup_steps}")
    for where, value in (("train.max_hours", t.max_hours), ("train.max_cost_usd", t.max_cost_usd)):
        if value is not None and value <= 0:
            errors.append(f"{where} must be > 0 or null, got {value}")

    choices = (
        ("data.source_type", d.source_type, SOURCE_TYPES),
        ("tokenizer.backend", tok.backend, TOKENIZER_BACKENDS),
        ("train.device", t.device, DEVICES),
        ("train.dtype", t.dtype, DTYPES),
    )
    for where, value, allowed in choices:
        if value not in allowed:
            errors.append(f"{where} must be one of {', '.join(allowed)}; got '{value}'")

    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))


def apply_overrides(raw: Optional[dict], overrides: Iterable[str]) -> dict:
    """Return a copy of raw with each 'section.key=value' override applied.

    Values are parsed as YAML, so 'true', '12', '3.0e-4' and 'null' get their natural types.
    """
    result = copy.deepcopy(raw) if raw else {}
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must look like section.key=value")
        dotted, value_text = item.split("=", 1)
        parts = dotted.strip().split(".")
        if len(parts) != 2 or not all(parts):
            raise ConfigError(
                f"override '{item}': key must be section.key, for example train.lr=3e-4"
            )
        section_name, key = parts
        section = result.get(section_name)
        if section is None:
            section = result[section_name] = {}
        if not isinstance(section, dict):
            raise ConfigError(f"override '{item}': section '{section_name}' is not a mapping")
        section[key] = yaml.safe_load(value_text)
    return result


def load_config(path: Union[str, Path], overrides: Iterable[str] = ()) -> Config:
    """Read a YAML config file, apply CLI overrides, and return a validated Config."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections")
    return config_from_dict(apply_overrides(raw, overrides))
