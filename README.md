# microgpt

A minimal, dependency-free GPT implementation in pure Python — train and run inference in a single file.

Based on [microgpt.py](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) by Andrej Karpathy.

## Usage

```bash
python microgpt.py
```

1000 stories are downloaded automatically from [karpathy/tinystories-gpt4-clean](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) on first run and cached in `input.txt`. After 1000 training steps the model generates 20 sample story snippets.

## How it works

1. **Dataset** — Load short stories from `input.txt` (downloaded automatically from HuggingFace if missing)
2. **Tokenizer** — Fixed 74-character ASCII vocabulary (as defined by the dataset) plus a special BOS token
3. **Autograd** — A minimal scalar `Value` class that tracks a computation graph for backpropagation
4. **Model parameters** — Initialise GPT weights: token/position embeddings, attention projections, MLP weights
5. **Forward pass** — For each token: embed → RMSNorm → multi-head attention → MLP → logits
6. **Training** — 1000 steps of forward pass, cross-entropy loss, backprop, and Adam weight updates
7. **Inference** — Sample 20 story snippets character-by-character using temperature-scaled softmax