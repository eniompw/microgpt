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

## How it works

1. **Dataset** — Load names from `input.txt` (downloaded automatically from GitHub if missing)
2. **Tokenizer** — Dynamic character vocabulary derived from the dataset plus a special BOS token
3. **Autograd** — A minimal scalar `Value` class that tracks a computation graph for backpropagation
4. **Model parameters** — Initialise GPT weights: token/position embeddings, attention projections, MLP weights
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — 1000 steps of forward pass, cross-entropy loss, backprop, and Adam weight updates
7. **Inference** — Sample 20 names character-by-character using temperature-scaled softmax