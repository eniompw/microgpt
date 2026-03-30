# microgpt

A minimal, dependency-free GPT implementation in pure Python — train and run inference in a single file.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

## Usage

```bash
python microgpt.py
```

~32,000 names are downloaded automatically from [karpathy/makemore](https://github.com/karpathy/makemore) on first run and cached in `input.txt`. After 1000 training steps the model generates 20 sample names.

## Colab notebook (PyTorch + GPU)

[microgpt-colab.ipynb](microgpt-colab.ipynb) is a PyTorch version designed to run on a free Colab T4 GPU.

Key differences from `microgpt.py`:

| | `microgpt.py` | `microgpt-colab.ipynb` |
|---|---|---|
| Backend | Pure Python (no deps) | PyTorch |
| Hardware | CPU | T4 GPU (Colab) |
| Dataset | Names (~32k, GitHub) | TinyStories (1000 stories, HuggingFace) |
| Vocabulary | Dynamic (from dataset) | Fixed 74-char ASCII |
| Output | Hallucinated names | Hallucinated story snippets |

> **Runtime:** Go to **Runtime → Change runtime type → T4 GPU** before running.

### Model configuration

| Hyperparameter | Value |
|---|---|
| `n_layer` | 4 |
| `n_embd` | 64 |
| `block_size` | 64 |
| `n_head` | 4 |
| Total parameters | ~214K |

### Estimated runtime (Colab T4 GPU)

| Section | Estimate |
|---|---|
| Setup & imports | < 5 s |
| Dataset download (10 API calls × 100 rows) | ~20–30 s (first run only) |
| Tokenizer + model init | < 1 s |
| **Training** (5000 steps × up to 64 tokens each, 64-dim model) | **~10–20 min on T4** |
| Inference (5 samples × ≤200 tokens) | < 10 s |

**Total: ~10–20 minutes on a Colab T4 GPU.**

- Training uses a Python `for` loop over tokens (no batching), which is the main bottleneck.
- Inference generates up to 200 tokens per sample, producing longer story snippets.
- After the first run, `input.txt` is cached so the download step is skipped.

## How it works

1. **Dataset** — Load names from `input.txt` (downloaded automatically from GitHub if missing)
2. **Tokenizer** — Dynamic character vocabulary derived from the dataset plus a special BOS token
3. **Autograd** — A minimal scalar `Value` class that tracks a computation graph for backpropagation
4. **Model parameters** — Initialise GPT weights: token/position embeddings, attention projections, MLP weights
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — 5000 steps of forward pass, cross-entropy loss, backprop, and Adam weight updates with linear LR decay
7. **Inference** — Sample 5 story snippets of up to 200 tokens using temperature-scaled softmax