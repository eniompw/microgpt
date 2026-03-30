# microgpt

A minimal, dependency-free GPT implementation in pure Python — train and run inference in a single file.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

## Usage

```bash
python microgpt.py
```

The dataset (`names.txt`) is downloaded automatically on first run. After 1000 training steps the model generates 20 sample names.

## How it works

1. **Dataset** — Load a list of names from `input.txt` (downloaded automatically if missing)
2. **Tokenizer** — Build a character-level vocabulary with a special BOS token
3. **Autograd** — A minimal scalar `Value` class that tracks a computation graph for backpropagation
4. **Model parameters** — Initialise GPT weights: token/position embeddings, attention projections, MLP weights
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — 1000 steps of forward pass, cross-entropy loss, backprop, and Adam weight updates
7. **Inference** — Sample 20 new names character-by-character using temperature-scaled softmax