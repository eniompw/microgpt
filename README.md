# microgpt

A minimal GPT trained from scratch — in a single Python file with no dependencies, or as a PyTorch Colab notebook with GPU support.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

---

## `microgpt.py` — Pure Python, no dependencies

Trains a tiny GPT on ~32,000 first names entirely in plain Python (no PyTorch, no NumPy).

```bash
python microgpt.py
```

`input.txt` is downloaded automatically from [karpathy/makemore](https://github.com/karpathy/makemore) on first run. After 1000 training steps the model generates 20 hallucinated names.

### How it works

1. **Dataset** — Character-level names loaded from `input.txt`
2. **Tokenizer** — Vocabulary built dynamically from the dataset, plus a special BOS token
3. **Autograd** — A minimal scalar `Value` class that builds a computation graph for backpropagation
4. **Model** — GPT weights: token/position embeddings, multi-head attention projections, MLP layers
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — Cross-entropy loss, backprop through the scalar graph, Adam updates with linear LR decay
7. **Inference** — Greedy or sampled decoding to generate new names

---

## `microgpt-colab.ipynb` — PyTorch + Colab T4 GPU

[microgpt-colab.ipynb](microgpt-colab.ipynb) is a PyTorch version of the same model, designed to run on a free Colab T4 GPU and trained on short story snippets instead of names.

> **Before running:** go to **Runtime → Change runtime type → T4 GPU**.

### Differences from `microgpt.py`

| | `microgpt.py` | `microgpt-colab.ipynb` |
|---|---|---|
| Backend | Pure Python | PyTorch |
| Hardware | CPU | T4 GPU (Colab) |
| Dataset | Names (~32k names, GitHub) | TinyStories (1000 stories, HuggingFace) |
| Vocabulary | Dynamic (from dataset) | Fixed 74-char ASCII |
| Output | Hallucinated names | Hallucinated story snippets |

### Default model configuration

| Hyperparameter | Value | Role |
|---|---|---|
| `n_layer` | 2 | Number of transformer blocks |
| `n_embd` | 16 | Embedding / hidden dimension |
| `block_size` | 16 | Max context length (tokens) |
| `n_head` | 4 | Number of attention heads |

### Estimated runtime (Colab T4 GPU, default config)

| Step | Time |
|---|---|
| Setup & imports | < 5 s |
| Dataset download (first run only) | ~20–30 s |
| Tokenizer + model init | < 1 s |
| Training (1000 steps) | ~2–4 min |
| Inference (5 samples × ≤200 tokens) | < 10 s |

Training uses a Python `for` loop over tokens with no batching — this is the main bottleneck. After the first run, `input.txt` is cached so the download is skipped.

### Scaling up: how hyperparameters affect training time

Each hyperparameter has a different cost profile:

| Hyperparameter | Cost scaling | Why |
|---|---|---|
| `num_steps` | linear | One forward+backward pass per step |
| `n_layer` | linear | Each layer adds the same fixed amount of compute |
| `n_embd` | quadratic | Weight matrices are `n_embd × n_embd` — doubling width ~4× cost |
| `block_size` | quadratic | Each token attends to all previous tokens — doubling context ~4× attention cost |

**Recommended order to scale up:** `num_steps` first (no memory cost), then `n_embd` + `block_size` together (biggest quality gain), then `n_layer`.

> Rule of thumb: doubling both `n_embd` and `block_size` increases training time by ~8–16×. Doubling only `n_layer` roughly doubles it.