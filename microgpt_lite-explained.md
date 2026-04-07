# microgpt_lite — Simplifications and Trade-offs

A walkthrough of every simplification made in `microgpt_lite.ipynb` relative to `microgpt_fast`, why each piece was safe to remove, and what was intentionally kept.

**Same model. Same training recipe. Less code.**

---

## Contents

- [microgpt_fast vs microgpt_lite](#microgpt_fast-vs-microgpt_lite) — side-by-side comparison of every change
- [Simplifications](#simplifications) — what was removed and why
  - [Single forward function](#single-forward-function)
  - [No KV cache](#no-kv-cache)
  - [Simplified apply_rope](#simplified-apply_rope)
  - [Condensed boilerplate](#condensed-boilerplate)
- [What stays](#what-stays) — the key speed improvements that were kept
- [Trade-offs](#trade-offs) — what's actually slower and by how much

---

## microgpt_fast vs microgpt_lite

| Aspect | `microgpt_fast` | `microgpt_lite` | Impact |
|---|---|---|---|
| Forward functions | 2 (`gpt_train` + `gpt`) | 1 (`forward`) | No speed impact |
| Inference method | KV cache | Re-run `forward` on sliding window | Slower inference only |
| `apply_rope` branches | 2 (batched + single-token) | 1 (batched) | No speed impact |
| Cells | ~15 | 8 | — |
| Import cells | 2 | 1 | — |
| `torch.compile` | ✓ | ✓ | ~2× speedup |
| Flash attention | ✓ | ✓ | ~30% speedup |
| RoPE | ✓ | ✓ | ~30% speedup |
| Mixed precision + GradScaler | ✓ | ✓ | ~2× throughput |
| Fused AdamW | ✓ | ✓ | small free speedup |
| Cosine LR + warmup + min_lr | ✓ | ✓ | better convergence |
| Gradient clipping | ✓ | ✓ | training stability |

Everything that matters for **training speed** is unchanged. The only real trade-off is inference speed, which is the least important part for a training demo.

---

## Simplifications

### Single forward function

`microgpt_fast` has two separate forward functions:

- `gpt_train(tokens)` — takes a `(B, T)` batch, returns `(B, T, vocab_size)` logits. Used during training.
- `gpt(token_id, pos_id, keys, values)` — takes a single token ID and position, maintains a KV cache across calls. Used during inference.

These two functions share the same attention + MLP logic but are implemented separately because they serve different shapes: one batched, one token-by-token with an explicit cache.

`microgpt_lite` replaces both with a single `forward(tokens)`:

```python
def forward(tokens):
    """tokens: (B, T) long → logits (B, T, vocab_size)"""
    B, T = tokens.shape
    x = rmsnorm(F.embedding(tokens, sd['wte']))
    ...
    return F.linear(rmsnorm(x), sd['wte'])
```

Training uses it exactly as before. Inference uses it with a sliding window — passing the last `block_size` tokens each step:

```python
def generate(max_tokens=200, temperature=0.7):
    tokens = [BOS]
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-block_size:]], device=device)
            logits = forward(x)[0, -1, :vocab_size]
            next_id = torch.multinomial(F.softmax(logits / temperature, -1), 1).item()
            if next_id == BOS: break
            tokens.append(next_id)
```

This is the standard, simple way to do autoregressive generation. The model re-processes the full context on every step instead of reusing cached intermediate values.

### No KV cache

The KV cache is an inference-only optimization. During training, every sequence in the batch is processed from scratch on every step — there's no re-use of past keys and values. Adding a KV cache only helps at inference time.

**What the KV cache does:** On each new token, instead of re-running attention over the entire context from scratch, the model caches the key and value tensors from previous tokens. This makes generation O(T) per token instead of O(T²) per token.

**Why it's safe to remove:** For the 5-sample × 200-token inference demo at the end of training, the difference is a few seconds. It matters in production models that generate thousands of tokens, not in a training demonstration.

**The cost:** Without the KV cache, generating 5 × 200 tokens re-runs the full sequence forward pass ~1000 times. In `microgpt_fast` this took ~7s; in `microgpt_lite` it takes longer — but this is after 3500 training steps, so the proportion of total runtime is small.

Removing the KV cache also eliminates the need for the single-token branch in `apply_rope` and the separate `gpt()` function entirely, which is the source of most of the code reduction.

### Simplified apply_rope

In `microgpt_fast`, `apply_rope` handles two shapes:

```python
def apply_rope(x, cos, sin):
    """x: (B,T,H,D) or (H,D). cos/sin: (T,D//2) or (D//2,)"""
    d = x.dtype
    x = x.float().unflatten(-1, (-1, 2))
    x_r, x_i = x[..., 0], x[..., 1]
    if x.dim() == 5:  # batched
        cos = cos.view(1, -1, 1, cos.shape[-1])
        sin = sin.view(1, -1, 1, sin.shape[-1])
    return torch.stack([x_r*cos - x_i*sin, x_r*sin + x_i*cos], -1).flatten(-2).to(d)
```

The `if x.dim() == 5` branch exists only to support the single-token `(H, D)` path used by the KV-cached `gpt()` function. Without the KV cache, queries and keys are always `(B, T, H, D)` — the batched path only.

`microgpt_lite` removes the branch entirely:

```python
def apply_rope(x, cos, sin):
    d = x.dtype
    x = x.float().unflatten(-1, (-1, 2))
    x0, x1 = x[..., 0], x[..., 1]
    cos = cos.view(1, -1, 1, cos.shape[-1])
    sin = sin.view(1, -1, 1, sin.shape[-1])
    return torch.stack([x0*cos - x1*sin, x0*sin + x1*cos], -1).flatten(-2).to(d)
```

The reshape is always applied, removing one conditional and making the function easier to follow.

### Condensed boilerplate

Several structural changes that reduce cells without affecting logic:

**Imports in one cell.** `microgpt_fast` imports `pandas` in cell 2 and `math` + `matplotlib` in cell 6. `microgpt_lite` puts all imports at the top:

```python
import os, random, json, math, time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
```

**Dataset + tokenizer in one cell.** The download, story loading, shuffle, character vocabulary, and encode/decode helpers are a single cell. None of these are logically separate — they're all data preparation.

**Shorter variable names in the weight dict.** `layer{i}.attn_wq` → `l{i}.wq`, `layer{i}.mlp_fc1` → `l{i}.fc1`. The layer structure is obvious from context.

**RoPE precomputation condensed.** Two intermediate variables (`t`, `f`) replace the named `freqs` variable — the computation reads left-to-right just as clearly:

```python
t = torch.arange(block_size, device=device).float()
f = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
rope_cos, rope_sin = torch.outer(t, f).cos(), torch.outer(t, f).sin()
```

---

## What stays

These are kept because they provide the bulk of the training speedup and are not complex to read:

**`torch.compile`** — one line applied to `forward` after definition. Fuses GPU kernels for ~2× speedup during training. The compiled function is called identically to the uncompiled version. You may see `Not enough SMs to use max_autotune_gemm mode` logged — this is informational only; the T4's 40 SMs fall below the threshold for exhaustive kernel search, but compilation still runs and fuses kernels normally.

**Flash attention** — `F.scaled_dot_product_attention(..., is_causal=True)`. One function call replaces an explicit attention matrix computation. No extra code required.

**RoPE** — two precomputed tensors (`rope_cos`, `rope_sin`) and one `apply_rope` call per layer. Fewer parameters than learned position embeddings and generally better.

**Mixed precision + GradScaler** — the `autocast` context manager and `scaler` wrapping add four lines to the training loop. Worth keeping: ~2× throughput on T4.

**Fused AdamW** — `fused=(device.type == 'cuda')` in the optimizer constructor. One keyword argument, free speedup.

**Cosine LR with warmup and min_lr** — `get_lr(step)` is 4 lines. Removing it and using a fixed LR would noticeably hurt convergence; the min_lr floor alone recovers ~0.17 loss (see `microgpt_fast-explained.md` Phase 6).

**Gradient clipping** — one line. Prevents occasional loss spikes at no cost.

---

## Trade-offs

| Trade-off | Magnitude | Matters? |
|---|---|---|
| Inference speed (no KV cache) | Several seconds per 5 samples | No — training dominates runtime |
| Code re-use (single `forward` for train + inference) | None | Simplification with no downside |
| `apply_rope` always reshapes `cos`/`sin` | Negligible | No — compile fuses it away |

The simplifications in `microgpt_lite` are all one-directional: they reduce code without touching training performance. The only real cost is inference speed, which is the least important part of this demonstration.
