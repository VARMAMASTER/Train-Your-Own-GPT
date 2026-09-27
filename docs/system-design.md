# Train Your Own GPT: System Design

Status: draft v1 · 2026-09-27

## 1. Goal

Build a GPT-style language model from scratch in PyTorch, train it, fine-tune it on custom data, and ship it as a public demo. Every core part (tokenizer, model, training loop, fine-tuning) is written by hand. Library code is used only for plumbing (downloading datasets, logging, the demo UI) and as a reference to test against.

**Success criteria**

1. A 124M-parameter model pretrained on at least 2.5B tokens, with a falling validation loss curve and coherent generated text.
2. The same model fine-tuned on custom data, with better validation loss on that data than the base model.
3. A measured comparison against a LoRA-tuned open model (Track B) on the same eval set.
4. A public GitHub repo with tests, a Hugging Face model card, and a Gradio demo anyone can try.
5. Total cloud spend within the $250 Vultr credit.

**Out of scope for v1:** RLHF/DPO, multi-node training, mixture-of-experts, quantization, serving at scale.

## 2. Constraints

| Constraint | Value | Design consequence |
|---|---|---|
| Local machine | Windows laptop, CPU only | All code must run on CPU with a tiny config for fast debugging. |
| Free GPU | Google Colab T4 (16 GB), sessions of at most 12h | Training must checkpoint often and resume exactly. |
| Paid GPU | Vultr, $250 credit | One script for all environments; budget guard that stops the run automatically. |
| Editor | VS Code + Google Colab extension | Colab runtime does not see local files; code reaches Colab through Git. |

## 3. Architecture overview

```mermaid
flowchart LR
    subgraph DATA["1 · Data"]
        RAW[Raw sources<br/>my data · TinyStories · FineWeb-Edu] --> CLEAN[Clean & dedupe]
        CLEAN --> SPLIT[Train / val split]
    end
    subgraph TOK["2 · Tokenizer"]
        BPE[Byte-level BPE<br/>train or load] --> ENC[Encode to uint16 shards]
    end
    subgraph MODEL["3 · Model"]
        GPT[GPT module]
    end
    subgraph TRAIN["4 · Training"]
        PRE[Pretrain] --> SFT[Fine-tune on my data]
    end
    subgraph EVAL["5 · Evaluation"]
        EV[Val loss · perplexity<br/>HellaSwag · samples]
    end
    subgraph SHIP["6 · Ship"]
        HUB[HF Hub weights] --> DEMO[Gradio demo]
    end
    SPLIT --> BPE
    SPLIT --> ENC
    ENC --> PRE
    GPT --> PRE
    SFT --> EV
    PRE --> EV
    EV --> HUB
    TRACKB[Track B · LoRA on open model] --> EV
```

Every stage reads files from the previous stage and writes files for the next. No stage calls another stage's internals. This keeps each stage testable alone and lets different stages run on different machines.

## 4. Repository layout

```
Train-Your-Own-GPT/
├── pyproject.toml            # package "tygpt", pip install -e .
├── configs/                  # YAML configs, one per experiment
│   ├── tiny_cpu.yaml         # ~2M params, laptop debugging
│   ├── small_tinystories.yaml# ~14M params, Colab
│   ├── base124m_fineweb.yaml # 124M params, Vultr
│   └── sft_mydata.yaml       # fine-tuning run
├── src/tygpt/
│   ├── config.py             # typed config dataclasses + YAML loading
│   ├── data/
│   │   ├── sources.py        # adapters: text files, JSONL, HF datasets
│   │   ├── clean.py          # normalisation, filtering, exact dedupe
│   │   └── prepare.py        # CLI: raw → cleaned → tokenized shards
│   ├── tokenizer/
│   │   ├── bpe.py            # byte-level BPE (train, encode, decode, save, load)
│   │   └── backends.py       # common interface: ScratchBPE | TiktokenGPT2
│   ├── model/
│   │   ├── gpt.py            # GPT, Block, CausalSelfAttention, MLP
│   │   └── generate.py       # sampling: temperature, top-k, top-p
│   ├── train/
│   │   ├── loader.py         # memmap token loader, SFT batch loader
│   │   ├── trainer.py        # training loop
│   │   ├── checkpoint.py     # save / resume
│   │   └── logging.py        # stdout, CSV, optional W&B
│   ├── eval/
│   │   ├── perplexity.py
│   │   ├── hellaswag.py
│   │   └── compare.py        # side-by-side report: scratch vs Track B
│   └── cli.py                # tygpt prepare | train | sample | eval
├── track_b/                  # LoRA fine-tune of an open model (HF + PEFT/Unsloth)
├── notebooks/
│   └── colab_runner.ipynb    # clone repo, install, run a config
├── scripts/                  # vultr_setup.sh, vultr_teardown.md checklist
├── demo/app.py               # Gradio app
├── tests/                    # pytest, runs on CPU
└── docs/
    ├── system-design.md      # this file
    └── results.md            # loss curves, tables, cost report
```

## 5. Components

Each component lists its job, its inputs and outputs, and the decisions that shape it.

### 5.1 Config

- **Job:** one typed source of truth for every run. No hard-coded hyperparameters anywhere else.
- **Shape:** Python dataclasses (`DataConfig`, `TokenizerConfig`, `ModelConfig`, `TrainConfig`, `PathsConfig`) loaded from YAML. Any field can be overridden on the command line: `tygpt train configs/small.yaml train.lr=3e-4`.
- **Why:** the same config file is saved inside every checkpoint, so any result can be reproduced exactly.

### 5.2 Data pipeline

- **Job:** turn raw sources into clean, split text.
- **Input:** a source spec in config, such as `type: text_dir`, `type: jsonl`, or `type: hf_dataset`.
- **Output:** `data/processed/<name>/{train,val}.txt` (or JSONL for chat data) plus `data_card.json` (document counts, bytes, filters applied, source hashes).
- **Steps:**
  1. **Load** through a source adapter. Each adapter yields plain documents. This is where your own data plugs in.
  2. **Clean:** Unicode NFC normalisation, strip control characters, drop documents that are too short or mostly non-text.
  3. **Deduplicate:** exact dedupe by SHA-256 of normalised text. Near-duplicate detection (MinHash) is deferred until real data shows a need.
  4. **Split** by document, not by line, with a fixed seed. Default is 99/1, and at least 1,000 val documents when available.
- **Big datasets:** FineWeb-Edu is streamed from Hugging Face and never fully held in memory.

### 5.3 Tokenizer

- **Job:** convert text to token IDs and back.
- **Interface** (both backends implement it):
  ```python
  class Tokenizer(Protocol):
      vocab_size: int
      def encode(self, text: str) -> list[int]: ...
      def decode(self, ids: list[int]) -> str: ...
      special_tokens: dict[str, int]   # <|endoftext|>, <|user|>, <|assistant|>, <|end|>
  ```
- **Backends:**
  - `ScratchBPE`: byte-level BPE written from scratch, with GPT-2-style regex pre-splitting. Trained on a sample of up to about 100 MB. Used for TinyStories and custom-data experiments. Default vocabulary is 8,192.
  - `TiktokenGPT2`: the GPT-2 vocabulary (50,257 tokens) through `tiktoken`. Used for the 124M FineWeb run, because encoding billions of tokens with a pure-Python encoder is too slow.
- **Proof that the scratch version is correct:** load GPT-2's published merges into `ScratchBPE` and check that its output matches `tiktoken` exactly on a test corpus. This test also shows interviewers that the scratch implementation is right, not just written.
- **Encoded format:** flat `uint16` NumPy arrays (`train.bin`, `val.bin`), with documents separated by `<|endoftext|>`. A sidecar `meta.json` records the tokenizer name, vocab size and token count. `uint16` works because both vocabularies are under 65,536.

### 5.4 Model

- **Job:** a decoder-only transformer, GPT-2 architecture.
- **Structure:** token embedding + learned position embedding → N × Block (pre-LayerNorm → causal multi-head self-attention → residual → pre-LayerNorm → MLP 4× with GELU → residual) → final LayerNorm → LM head with weights tied to the token embedding.
- **Attention:** uses `torch.nn.functional.scaled_dot_product_attention(is_causal=True)`, which picks flash attention on supported GPUs. A manual implementation is kept too, and a test checks that both give the same output.
- **Initialisation:** normal(0, 0.02), with residual projections scaled by 1/√(2·n_layer) as in GPT-2.
- **Presets:**

| Preset | Layers | Heads | d_model | Context | Vocab | ~Params | Runs on |
|---|---|---|---|---|---|---|---|
| `tiny` | 4 | 4 | 128 | 256 | 8,192 | 2M | Laptop CPU |
| `small` | 6 | 6 | 384 | 512 | 8,192 | 14M | Colab T4 |
| `base124m` | 12 | 12 | 768 | 1,024 | 50,304* | 124M | Vultr A100/H100 |

\*50,257 padded to a multiple of 64 for GPU efficiency.

### 5.5 Training

- **Job:** one trainer for pretraining and fine-tuning, on any device.
- **Loop features:**
  - AdamW (β = 0.9, 0.95; weight decay 0.1 on 2-D weights only)
  - Linear warmup then cosine decay to 10% of the peak learning rate
  - Gradient clipping at 1.0
  - Gradient accumulation, so the effective batch size doesn't depend on GPU memory
  - Mixed precision: bf16 on A100/H100, fp16 with GradScaler on T4, fp32 on CPU
  - `torch.compile` on when the environment supports it
  - Evaluation every `eval_interval` steps on a fixed val subset
- **Device selection:** automatic (`cuda` → `mps` → `cpu`), with the precision chosen to match.
- **Scale:** single GPU for v1. Multi-GPU DDP is a later extension. The trainer is written so it can be added without restructuring.

### 5.6 Checkpointing and resume

- **Saved every `ckpt_interval` steps and on exit:** model weights, optimizer state, GradScaler state, step, config, RNG states, data-loader position, and the best val loss so far.
- **Two files are kept:** `last.pt`, which is overwritten, and `best.pt`, the lowest val loss. Writes are atomic (write to a temporary file, then rename), so a disconnect mid-write can't corrupt a checkpoint.
- **Resume:** `tygpt train <config> --resume` continues from `last.pt`. A test checks that "train 20 steps" and "train 10 steps, stop, resume, train 10 more" produce the same weights.

### 5.7 Fine-tuning (SFT)

The fine-tuning mode depends on your data type:

| Data type | Mode | How it works |
|---|---|---|
| Documents / plain text | **Continued pretraining** | Same loss as pretraining on the new text, with a lower learning rate. |
| Conversations / Q&A | **Supervised fine-tuning** | JSONL `{"messages":[{"role","content"},...]}` rendered with special tokens. Loss is computed only on assistant tokens (a mask sets user-token targets to -100). |

In both modes, training starts from the pretrained checkpoint, uses a learning rate about 10× lower, and runs a few epochs with early stopping on custom-data val loss, so the model doesn't just memorise the data.

### 5.8 Generation

- `generate(model, prompt_ids, max_new_tokens, temperature, top_k, top_p)`, stopping at `<|end|>` or `<|endoftext|>`.
- v1 recomputes the full context at every step. A KV cache is a later optimisation, and a test will check it gives the same output as the no-cache version.

### 5.9 Evaluation

| Metric | Applies to | Purpose |
|---|---|---|
| Val loss / perplexity | All runs | Main training signal |
| Fixed-prompt samples | All runs | Qualitative check, saved at every eval |
| HellaSwag accuracy | `base124m` | Standard benchmark. The GPT-2 124M reference is ~29.5% |
| Custom-data val loss | Scratch SFT vs Track B | The head-to-head comparison |

`compare.py` writes `docs/results.md` with tables and loss-curve PNGs.

### 5.10 Track B (LoRA baseline)

- Lives in `track_b/` and deliberately uses libraries: Hugging Face `transformers`, `peft` or Unsloth, and a 1B–3B open model (Qwen, Llama or Gemma).
- It uses the **same processed dataset and the same val split** as the scratch model (section 5.2), so the comparison is fair.
- It runs on Colab with QLoRA.

### 5.11 Demo

- `demo/app.py`: a Gradio chat or completion UI that loads weights from the Hugging Face Hub. It runs on a free HF Spaces CPU instance, where a 124M model is fast enough.
- Model card: architecture, data, training compute and cost, eval results, limitations.

## 6. Environments and workflow

The same code and configs run everywhere. Only `paths.*` and the device change.

```mermaid
flowchart LR
    LAP[Laptop · VS Code<br/>write code · tiny config · tests] -->|git push| GH[(GitHub repo)]
    GH -->|git clone / pull| COLAB[Colab runtime<br/>small runs · SFT · Track B]
    GH -->|git clone| VULTR[Vultr GPU server<br/>base124m pretraining]
    COLAB -->|checkpoints| DRIVE[(Google Drive)]
    VULTR -->|checkpoints| OBJ[(Vultr Object Storage)]
    DRIVE --> HF[(Hugging Face Hub)]
    OBJ --> HF
```

| Environment | How code arrives | Data & checkpoint location | Used for |
|---|---|---|---|
| Laptop | Local working copy | `./data`, `./checkpoints` | Coding, unit tests, `tiny` config |
| Colab (from VS Code) | `colab_runner.ipynb` runs `git clone` + `pip install -e .` | `/content/drive/MyDrive/tygpt/` | `small` runs, SFT, Track B |
| Vultr | `git clone` + setup script | Local NVMe, synced to object storage every checkpoint | `base124m` pretraining |

**Important:** the VS Code Colab extension runs notebook cells on Colab, but the Colab machine does **not** see files on your laptop. Code gets there through GitHub, and data gets there through Drive or a Hugging Face download.

## 7. Failure handling

| Failure | Detection | Response |
|---|---|---|
| Colab disconnect / timeout | Session ends | Resume from `last.pt` on Drive. At most `ckpt_interval` steps are lost. |
| Loss becomes NaN/Inf | Checked every step | Save a debug checkpoint, log the last batch index, stop with a clear error. |
| Loss spike | Loss > 3× running average | Log a warning. Stop if it persists for more than N evals. |
| CUDA out of memory | Exception on first steps | Error message suggests halving `micro_batch_size`; accumulation keeps the effective batch the same. |
| Vultr overspend | `train.max_hours` and `train.max_cost_usd` in config | Trainer saves and exits cleanly when either limit is reached. |
| Forgetting to destroy the Vultr instance | Human process | `scripts/vultr_teardown.md` checklist: sync checkpoints → verify → destroy. |
| Corrupt checkpoint | Atomic write (temp file + rename) | The previous checkpoint is always valid. |

## 8. Testing strategy

All tests run on CPU in under about 2 minutes, locally and in GitHub Actions on every push.

| Area | Tests |
|---|---|
| Tokenizer | encode→decode round trip on random Unicode; ScratchBPE with GPT-2 merges matches tiktoken exactly; special tokens never split |
| Data | dedupe removes exact duplicates; split has no document in both train and val; `uint16` shards round-trip |
| Model | output shapes; parameter count matches preset; **causality** (changing token t+1 never changes logits at ≤ t); SDPA and manual attention agree |
| Training | overfits a single batch to near-zero loss; save → resume gives the same weights as uninterrupted training; SFT mask gives zero loss contribution from user tokens |
| Generation | deterministic with fixed seed; stops at end token |

## 9. Compute and budget plan

Training compute is estimated as FLOPs ≈ 6 × parameters × tokens.

| Run | Params | Tokens | FLOPs | Hardware | Est. time | Est. cost |
|---|---|---|---|---|---|---|
| `tiny` debug | 2M | 5M | 6e13 | Laptop | minutes | $0 |
| `small` TinyStories | 14M | ~470M | 4e16 | Colab T4 | 1–3 h | $0 |
| `base124m` FineWeb-Edu | 124M | 2.5B | 1.9e18 | 1× A100 80GB | ~4–6 h | ~$10–25 |
| `base124m` extended | 124M | 10B | 7.4e18 | 1× A100/H100 | ~12–20 h | ~$30–60 |

Estimates assume about 35–40% of peak GPU throughput. **Before the real run, a 15-minute benchmark on the rented GPU measures actual tokens/sec**, and the final token budget is set from that number, not from this table.

**Budget allocation ($250):** benchmark and setup ~$15 · main 124M run ~$25 · extended or 350M stretch run ≤ $100 · reserve for reruns ≥ $100.

## 10. Build order

Each step is its own small project with its own plan, tests and PR. Each produces something runnable.

| # | Sub-project | Done when | Runs on |
|---|---|---|---|
| 0 | Scaffold: package, config, CLI, CI | `pytest` passes in GitHub Actions | Laptop |
| 1 | Data pipeline | TinyStories + custom data → split text with data card | Laptop |
| 2 | Tokenizer | ScratchBPE matches tiktoken; shards written | Laptop |
| 3 | Model + generation | All model tests pass; `tiny` overfits a batch | Laptop |
| 4 | Trainer + checkpointing | `tiny` trains end-to-end on CPU; resume test passes | Laptop |
| 5 | Colab runner | `small` trains on TinyStories, survives a forced disconnect | Colab |
| 6 | Evaluation | Perplexity, samples, HellaSwag working | Colab |
| 7 | Big run | `base124m` trained within budget; results logged | Vultr |
| 8 | Fine-tuning | Custom-data val loss improves over base | Colab |
| 9 | Track B + comparison | `results.md` with head-to-head table | Colab |
| 10 | Ship | HF Hub weights, Gradio Space live, README with results | HF |

## 11. Dependencies

| Package | Why | Where |
|---|---|---|
| `torch` | Model and training | Core |
| `numpy` | Token shards (memmap) | Core |
| `pyyaml` | Configs | Core |
| `tqdm` | Progress bars | Core |
| `datasets` | Stream TinyStories / FineWeb-Edu | Data |
| `tiktoken` | GPT-2 tokenizer backend + reference for tests | Tokenizer |
| `pytest` | Tests | Dev |
| `wandb` (optional) | Loss dashboards | Training |
| `transformers`, `peft` / `unsloth` | Track B only | `track_b/` |
| `gradio`, `huggingface_hub` | Demo and publishing | `demo/` |

## 12. Open decisions

| Decision | Options | Blocks |
|---|---|---|
| **Custom data type** | Documents · conversations/Q&A · code · undecided | Step 1 source adapter, step 8 fine-tuning mode (section 5.7) |
| Custom data size | MB · hundreds of MB · GB | Whether ScratchBPE is retrained on it, and the number of SFT epochs |
| Experiment tracking | W&B (free tier) · CSV + matplotlib only | Step 4 logging |
