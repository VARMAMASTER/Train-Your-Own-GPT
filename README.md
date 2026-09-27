# Train Your Own GPT

A GPT-style language model built from scratch in PyTorch: BPE tokenizer, transformer, training loop, and fine-tuning on custom data.

Development runs locally, experiments on Google Colab (via the VS Code Colab extension), and the large pretraining run on a Vultr GPU server.

Status: step 0 (scaffold) complete. Next: step 1, the data pipeline.

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
