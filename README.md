# microgpt

A minimal GPT trained from scratch — in a single Python file with no dependencies, or as a PyTorch Colab notebook with GPU support.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

The updated `microgpt_fast.ipynb` and `microgpt_fast.py` took significant inspiration from [`model.py`](https://github.com/EN10/modded-llama2.c/blob/main/model.py) and [`train.py`](https://github.com/EN10/modded-llama2.c/blob/main/train.py) from [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c).

---

## `microgpt.py` — Pure Python, no dependencies

Trains a tiny GPT on ~32,000 first names entirely in plain Python (no PyTorch, no NumPy).

```bash
python microgpt.py
```

`input.txt` is downloaded automatically from [karpathy/makemore](https://github.com/karpathy/makemore) on first run. After 1000 training steps the model generates 20 hallucinated names.

See [microgpt-explained.md](microgpt-explained.md) for a detailed walkthrough of every component — autograd, attention, training loop, and inference.

---

## `microgpt_torch.py` — PyTorch port of `microgpt.py`

A direct PyTorch translation of `microgpt.py`. Same dataset (names), same hyperparameters, same single-sequence training loop — but replaces the hand-rolled autograd `Value` class with PyTorch tensors and `nn.Module`.

Useful for understanding the exact mapping from the pure-Python code to idiomatic PyTorch.

```bash
python microgpt_torch.py
```

---

## `microgpt_fast.py` — Semi-optimised PyTorch (GPU)

A standalone Python script with the same architecture and training changes as the Colab notebook (see table below), but runnable outside of Colab. Trains on the TinyStories dataset with batched processing, mixed precision, RoPE, flash attention, weight tying, and `torch.compile`.

```bash
python microgpt_fast.py
```

> Requires a CUDA GPU. Falls back to CPU but will be significantly slower.

---

## `microgpt_fast.ipynb` — PyTorch + Colab T4 GPU

[microgpt_fast.ipynb](microgpt_fast.ipynb) is a PyTorch version of the same model, designed to run on a free Colab T4 GPU and trained on short story snippets instead of names.

> **Before running:** go to **Runtime → Change runtime type → T4 GPU**.

### Differences from `microgpt.py`

| | `microgpt.py` | `microgpt_fast.ipynb` |
|---|---|---|
| Backend | Pure Python | PyTorch |
| Hardware | CPU | T4 GPU (Colab) |
| Dataset | Names (~32k names, GitHub) | TinyStories (5000 stories, HuggingFace) |
| Vocabulary | Dynamic (from dataset) | Fixed 74-char ASCII |
| Output | Hallucinated names | Hallucinated story snippets |
| Training time | ~5–15 min (CPU) | ~2–5 min (T4 GPU) |

### Model configuration

| Hyperparameter | Value | Role |
|---|---|---|
| `n_layer` | 5 | Number of transformer blocks |
| `n_embd` | 128 | Embedding / hidden dimension |
| `block_size` | 256 | Max context length (tokens) |
| `n_head` | 8 | Number of attention heads |
| `batch_size` | 128 | Sequences per gradient step |
| `num_steps` | 2000 | Training steps |

### Architecture and training changes (inspired by [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c))

The updated notebook and `microgpt_fast.py` adopt a Llama-style design in place of the original minimal GPT:

Relative improvements below are informed by the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), where each technique was measured in isolation on 8×H100 GPUs.

**Architecture**

| Change | Benefit | Relative improvement (modded-nanogpt) |
|---|---|---|
| **Batched sequence processing** — full `(B, T)` tensor forward pass instead of token-by-token | Massive GPU utilization; enables all other speed gains | Foundational — prerequisite for all records |
| **Flash attention** — `F.scaled_dot_product_attention` with `is_causal=True` | Fused O(T) memory attention kernel | ~30% faster (#12: 7.2 → 5.0 min) |
| **RoPE** (Rotary Position Embeddings) instead of learned position embeddings | Better length generalization, no extra parameters | ~30% faster (#2: 45 → 31 min) |
| **Weight tying** — embedding matrix reused as lm_head | Fewer parameters, stronger gradient signal to embeddings | ~neutral speed, saves params (#8/#51) |
| **RMSNorm** instead of LayerNorm (pre-norm) | Simpler, faster, no mean subtraction or bias | ~5–10% faster (standard in all records) |
| **KV cache** — per-layer key/value cache at inference | Efficient token-by-token generation (linear instead of quadratic) | Inference-only (not measured in speedrun) |

**Training**

| Change | Benefit | Relative improvement (modded-nanogpt) |
|---|---|---|
| **Mixed precision** — `float16` autocast + `GradScaler` | ~2× speed, halves memory bandwidth; enables larger batches on T4 | ~5% faster (#10: 8.2 → 7.8 min) |
| **AdamW** optimizer with linear learning rate decay (matching `microgpt.py`) | Decoupled weight decay for better generalization | Foundational — used in all records |
| **`torch.compile`** — fuses GPU kernels on training forward pass | ~2× additional speedup by optimizing the computation graph | ~8% faster (#7: 13.1 → 12.0 min) |