# microgpt

A minimal GPT trained from scratch — designed to teach LLM development from first principles. Start with a pure-Python implementation, then progress to PyTorch, then to GPU-optimised training on a free Colab T4.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy, with GPU training inspired by [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c).

---

## Learning path

| Step | File | What you learn |
|---|---|---|
| 1 | `microgpt.py` | How a GPT works from scratch — autograd, attention, training loop, inference — with zero dependencies |
| 2 | `microgpt_torch.py` | How the same model maps to PyTorch (`nn.Module`, tensors, autograd) |
| 3 | `microgpt_fast.ipynb` / `.py` | How to make it actually work — batched training, GPU acceleration, modern transformer techniques |
| 4 | `microgpt_lite.ipynb` | How to strip back complexity while keeping the key speed wins |

---

## `microgpt.py` — Pure Python, no dependencies

Trains a tiny GPT on ~32,000 first names entirely in plain Python (no PyTorch, no NumPy).

```bash
python microgpt.py
```

`input.txt` is downloaded automatically from [karpathy/makemore](https://github.com/karpathy/makemore) on first run. After 1000 training steps the model generates 20 hallucinated names.

See [microgpt-explained.md](microgpt-explained.md) for a detailed walkthrough of every component — autograd, attention, training loop, and inference.

---

## `microgpt_torch.py` — PyTorch port

A direct PyTorch translation of `microgpt.py`. Same dataset (names), same hyperparameters, same single-sequence training loop — but replaces the hand-rolled autograd `Value` class with PyTorch tensors and `nn.Module`.

Useful for understanding the exact mapping from the pure-Python code to idiomatic PyTorch.

```bash
python microgpt_torch.py
```

---

## `microgpt_fast` — GPU-optimised PyTorch

Available as a Colab notebook (`microgpt_fast.ipynb`) or standalone script (`microgpt_fast.py`) — same model and training code in both.

Trains a Llama-style transformer on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset (~5000 short stories) using batched processing, mixed precision, flash attention, RoPE, weight tying, and `torch.compile`. Generates hallucinated children's stories.

### Running

**Colab notebook** (recommended — free GPU):
1. Open [microgpt_fast.ipynb](microgpt_fast.ipynb) in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells

**Local script** (requires a CUDA GPU):
```bash
python microgpt_fast.py
```

### Differences from `microgpt.py`

| | `microgpt.py` | `microgpt_fast` |
|---|---|---|
| Backend | Pure Python | PyTorch |
| Hardware | CPU | T4 GPU (Colab) |
| Dataset | Names (~32k names, GitHub) | TinyStories (5000 stories, HuggingFace) |
| Vocabulary | Dynamic (from dataset) | Fixed 74-char ASCII |
| Output | Hallucinated names | Hallucinated story snippets |
| Training time | ~5–15 min (CPU) | ~3 min (T4 GPU) |

### Model configuration

| Hyperparameter | Value | Role |
|---|---|---|
| `n_layer` | 6 | Number of transformer blocks |
| `n_embd` | 256 | Embedding / hidden dimension |
| `block_size` | 256 | Max context length (tokens) |
| `n_head` | 8 | Number of attention heads |
| `batch_size` | 64 | Sequences per gradient step |
| `num_steps` | 3500 | Training steps |

### Architecture and training

Uses a Llama-style transformer with RMSNorm, RoPE, flash attention, SiLU, weight tying, and KV-cached inference. Training uses mixed precision, `torch.compile`, cosine LR with warmup and min_lr floor, gradient clipping, and AdamW.

See [microgpt_fast-explained.md](microgpt_fast-explained.md) for a detailed breakdown of every design choice, the optimization journey from broken output to coherent stories, a hyperparameter tuning guide, and a glossary of all technical terms.

---

## `microgpt_lite` — Simplified GPU PyTorch

A streamlined version of `microgpt_fast` — same architecture and training recipe, but reduced to the minimum code needed to understand and run it.

**Open in Colab** (recommended — free GPU):
1. Open [microgpt_lite.ipynb](microgpt_lite.ipynb) in Google Colab
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells

### What's removed compared to `microgpt_fast`

| Removed | Reason |
|---|---|
| KV cache + separate `gpt()` inference function | Single `forward()` reused for both training and inference |
| `apply_rope` single-token branch | Batched path only — no separate code path needed |
| Separate dataset/tokenizer cells with verbose prints | Condensed to one cell |
| Imports spread across cells | All imports in one cell |

### What stays (the important parts)

- `torch.compile` — ~2× kernel fusion speedup
- Flash attention (`scaled_dot_product_attention`) — fused CUDA kernel
- RoPE — rotary position embeddings, no extra parameters
- Mixed precision + `GradScaler` — float16 forward, float32 gradients
- Fused AdamW — single kernel optimizer step on CUDA
- Cosine LR with warmup + `min_lr` floor
- Gradient clipping

The only meaningful speed trade-off is inference: without a KV cache each generated token re-runs the full sequence through `forward()`, which is O(T²) per sample instead of O(T). This is negligible for the 5-sample demo at the end of training.

See [microgpt_lite-explained.md](microgpt_lite-explained.md) for a walkthrough of every simplification and why the removed pieces were safe to drop.