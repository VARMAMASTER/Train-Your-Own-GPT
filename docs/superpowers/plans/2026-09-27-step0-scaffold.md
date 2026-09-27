# Step 0: Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An installable `tygpt` Python package with a typed YAML config system, dotted CLI overrides, a `tygpt config` command, a pytest suite, and GitHub Actions CI that runs the tests on CPU.

**Architecture:** src-layout package (`src/tygpt/`). `config.py` defines one dataclass per config section (`data`, `tokenizer`, `model`, `train`, `paths`) plus a root `Config`. YAML is loaded into a plain dict, `section.key=value` overrides are applied to that dict, then the dict is converted to dataclasses with type coercion and validation. `cli.py` is an argparse front end. Later steps add their own subcommands to it.

**Tech Stack:** Python ≥ 3.10, setuptools, PyYAML, pytest, GitHub Actions. torch/numpy/tqdm are declared as dependencies now but not used until later steps.

**Spec:** `docs/system-design.md` (sections 4, 5.1, 8, 10 step 0)

## Global Constraints

- Package name: `tygpt`; source under `src/tygpt/`; tests under `tests/`.
- `requires-python = ">=3.10"` (Colab and the local machine run 3.13; CI tests 3.10 and 3.13).
- Hyperparameters live only in config; no hard-coded values elsewhere (spec 5.1).
- CLI override syntax is exactly `section.key=value`, e.g. `train.lr=3e-4` (spec 5.1).
- Token shards are `uint16`, so `tokenizer.vocab_size` must be ≤ 65,536 (spec 5.3).
- Allowed values: `data.source_type` ∈ {text_dir, jsonl, hf_dataset}; `tokenizer.backend` ∈ {scratch_bpe, tiktoken_gpt2}; `train.device` ∈ {auto, cpu, cuda, mps}; `train.dtype` ∈ {auto, float32, bfloat16, float16}.
- All tests run on CPU (spec 8).
- Windows dev machine: run tools through `.venv\Scripts\python -m ...` so PowerShell execution policy never blocks venv activation.
- Every commit message ends with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, console script, pytest settings |
| `src/tygpt/__init__.py` | Package version (single source of truth) |
| `src/tygpt/config.py` | Config dataclasses, coercion, validation, YAML loading, overrides |
| `src/tygpt/cli.py` | `tygpt` command: argparse parser and `config` subcommand |
| `configs/tiny_cpu.yaml` | ~2M-param laptop debugging config |
| `tests/test_package.py` | Package imports and exposes version |
| `tests/test_config.py` | Dict → Config conversion, coercion, validation |
| `tests/test_config_loading.py` | YAML files, overrides, every shipped config loads |
| `tests/test_cli.py` | CLI behaviour and exit codes |
| `.github/workflows/ci.yml` | Run pytest on CPU for Python 3.10 and 3.13 |
| `README.md` | Add development setup section |

---

### Task 1: Installable package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/tygpt/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `tygpt` with `tygpt.__version__ == "0.1.0"`; a working `.venv` with the package installed in editable mode plus pytest.

- [ ] **Step 1: Create the virtualenv**

Run (PowerShell, repo root):
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
```
Expected: `.venv\` exists (already covered by `.gitignore`).

- [ ] **Step 2: Write the failing test**

`tests/test_package.py`:
```python
import tygpt


def test_package_exposes_version():
    assert tygpt.__version__ == "0.1.0"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\python -m pip install pytest; .venv\Scripts\python -m pytest tests/test_package.py`
Expected: FAIL / collection error with `ModuleNotFoundError: No module named 'tygpt'`.

- [ ] **Step 4: Write the package files**

`src/tygpt/__init__.py`:
```python
"""Train Your Own GPT: a GPT-style language model built from scratch."""

__version__ = "0.1.0"
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "tygpt"
dynamic = ["version"]
description = "A GPT-style language model built from scratch in PyTorch"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.dynamic]
version = { attr = "tygpt.__version__" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 5: Install in editable mode and run the test**

Run: `.venv\Scripts\python -m pip install -e ".[dev]"; .venv\Scripts\python -m pytest tests/test_package.py`
Expected: `1 passed`. (The first install downloads torch, which takes a few minutes.)

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml src/tygpt/__init__.py tests/test_package.py
git commit -m "feat: add installable tygpt package skeleton" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Config dataclasses, coercion and validation

**Files:**
- Create: `src/tygpt/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces (used by Task 3, Task 4 and every later step):
  - `class ConfigError(ValueError)`
  - dataclasses `DataConfig`, `TokenizerConfig`, `ModelConfig`, `TrainConfig`, `PathsConfig`, `Config` (fields exactly as in Step 3)
  - `config_from_dict(raw: dict | None) -> Config`: coerces types, rejects unknown sections/keys, calls `validate`
  - `config_to_dict(cfg: Config) -> dict`
  - `validate(cfg: Config) -> None`: raises `ConfigError` listing every problem

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_config.py`
Expected: collection error `ModuleNotFoundError: No module named 'tygpt.config'`.

- [ ] **Step 3: Write the implementation**

`src/tygpt/config.py`:
```python
"""Typed run configuration: one dataclass per section, loaded from plain dicts."""

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Optional

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_config.py`
Expected: all tests pass (25 including parametrized cases).

- [ ] **Step 5: Commit**

```powershell
git add src/tygpt/config.py tests/test_config.py
git commit -m "feat: add typed config dataclasses with coercion and validation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: YAML loading, CLI overrides and the tiny config

**Files:**
- Modify: `src/tygpt/config.py` (add imports and two functions at the end)
- Create: `configs/tiny_cpu.yaml`
- Test: `tests/test_config_loading.py`

**Interfaces:**
- Consumes: `Config`, `ConfigError`, `config_from_dict` from Task 2.
- Produces:
  - `apply_overrides(raw: dict | None, overrides: Iterable[str]) -> dict`: returns a new dict; input untouched
  - `load_config(path: str | Path, overrides: Iterable[str] = ()) -> Config`
  - `configs/tiny_cpu.yaml`, which later steps use as the laptop debugging run

- [ ] **Step 1: Write the failing tests**

`tests/test_config_loading.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_config_loading.py`
Expected: collection error `ImportError: cannot import name 'apply_overrides' from 'tygpt.config'`.

- [ ] **Step 3: Add the implementation**

In `src/tygpt/config.py`, replace the import block at the top with:
```python
import copy
import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import yaml
```

Append to the end of `src/tygpt/config.py`:
```python
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
```

Create `configs/tiny_cpu.yaml`:
```yaml
# ~2M-parameter model for fast debugging on a laptop CPU.
data:
  name: tinystories
  source_type: hf_dataset
  source: roneneldan/TinyStories

tokenizer:
  backend: scratch_bpe
  vocab_size: 8192

model:
  n_layer: 4
  n_head: 4
  d_model: 128
  block_size: 256
  vocab_size: 8192

train:
  max_steps: 200
  micro_batch_size: 8
  lr: 1.0e-3
  warmup_steps: 20
  eval_interval: 50
  eval_steps: 10
  ckpt_interval: 100
  device: cpu
  dtype: float32

paths:
  data_dir: data
  ckpt_dir: checkpoints/tiny_cpu
```

- [ ] **Step 4: Run the full suite**

Run: `.venv\Scripts\python -m pytest`
Expected: all tests in `test_package.py`, `test_config.py` and `test_config_loading.py` pass.

- [ ] **Step 5: Commit**

```powershell
git add src/tygpt/config.py configs/tiny_cpu.yaml tests/test_config_loading.py
git commit -m "feat: load YAML configs with section.key=value overrides; add tiny_cpu config" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `tygpt` command-line entry point

**Files:**
- Create: `src/tygpt/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]` after `[project.optional-dependencies]`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `load_config`, `config_to_dict`, `ConfigError` from Tasks 2–3; `tygpt.__version__`.
- Produces:
  - `build_parser() -> argparse.ArgumentParser`: later steps register subcommands here (`prepare`, `train`, `sample`, `eval`), each with `set_defaults(func=...)`
  - `main(argv: list[str] | None = None) -> int`: returns 0 on success, 1 on `ConfigError`
  - console script `tygpt`

- [ ] **Step 1: Write the failing tests**

`tests/test_cli.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_cli.py`
Expected: collection error `ModuleNotFoundError: No module named 'tygpt.cli'`.

- [ ] **Step 3: Write the implementation**

`src/tygpt/cli.py`:
```python
"""Command-line entry point: `tygpt <command> ...`."""

import argparse
import sys
from typing import Optional

import yaml

from tygpt import __version__
from tygpt.config import ConfigError, config_to_dict, load_config


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, args.overrides)
    print(yaml.safe_dump(config_to_dict(cfg), sort_keys=False), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tygpt", description="Train Your Own GPT")
    parser.add_argument("--version", action="version", version=f"tygpt {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    config = commands.add_parser(
        "config", help="load a config, apply overrides and print the resolved result"
    )
    config.add_argument("config", help="path to a YAML config file")
    config.add_argument("overrides", nargs="*", help="overrides like train.lr=3e-4")
    config.set_defaults(func=cmd_config)

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

In `pyproject.toml`, insert after the `[project.optional-dependencies]` block:
```toml
[project.scripts]
tygpt = "tygpt.cli:main"
```

- [ ] **Step 4: Run tests, reinstall, and smoke-test the console script**

Run:
```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m pip install -e ".[dev]" --quiet
.venv\Scripts\tygpt config configs\tiny_cpu.yaml train.lr=5e-4
.venv\Scripts\tygpt config configs\tiny_cpu.yaml train.nope=1; echo "exit=$LASTEXITCODE"
```
Expected: all tests pass; first command prints YAML containing `lr: 0.0005`; second prints `error: unknown key 'train.nope'...` and `exit=1`.

- [ ] **Step 5: Commit**

```powershell
git add src/tygpt/cli.py pyproject.toml tests/test_cli.py
git commit -m "feat: add tygpt CLI with config command" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: GitHub Actions CI and development docs

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md` (append "Development" section)

**Interfaces:**
- Consumes: `pyproject.toml` `dev` extra and the full test suite from Tasks 1–4.
- Produces: CI that runs `pytest` on every push to `main` and every pull request, on Python 3.10 and 3.13, with CPU-only torch.

- [ ] **Step 1: Write the workflow**

`.github/workflows/ci.yml`:
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.13"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
      - name: Install CPU-only PyTorch
        run: pip install torch --index-url https://download.pytorch.org/whl/cpu
      - name: Install package
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest
```

- [ ] **Step 2: Add development docs**

Append to `README.md`:
````markdown

## Development

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Inspect a config with overrides applied:

```powershell
.venv\Scripts\tygpt config configs\tiny_cpu.yaml train.lr=5e-4
```

Design: [docs/system-design.md](docs/system-design.md) · visual version: [docs/system-design.html](docs/system-design.html)
````

- [ ] **Step 3: Run the full suite locally one last time**

Run: `.venv\Scripts\python -m pytest`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```powershell
git add .github/workflows/ci.yml README.md
git commit -m "ci: run tests on CPU for Python 3.10 and 3.13; add dev setup docs" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Push and confirm CI is green (ask the user before pushing)**

Run: `git push origin main`, then `gh run watch` (or check the Actions tab on GitHub).
Expected: both matrix jobs (3.10, 3.13) pass. If a job fails, read the log, fix the cause, commit and push again.
