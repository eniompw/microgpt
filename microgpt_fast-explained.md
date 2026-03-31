# microgpt_fast — Architecture and Training Choices

A detailed breakdown of every design decision in `microgpt_fast.ipynb` / `microgpt_fast.py`, why it was chosen, and how much it helps.

**Training time: ~3 min on a T4 GPU** (3500 steps, 4M tokens, ~5M parameters).

---

## Optimization journey

The project started as a direct PyTorch port of the pure-Python `microgpt.py` — token-by-token training on a tiny model. Most of the iteration from there was trying to get loss low enough that the output reads as coherent English, first by overhauling the architecture, then by tuning the training recipe and model size.

### Phase 1: PyTorch port (token-by-token)

The initial Colab notebook was a straight conversion of `microgpt.py` to PyTorch — same token-by-token training loop, same tiny model.

| Commit | Config | What changed |
|---|---|---|
| `9857ea2` | 1 layer, 16-dim, block=16, lr=0.01, 1000 steps | Initial port — ReLU, learned positions, separate lm_head |
| `3843e16` | 2 layers, 32-dim, block=32 | Tried scaling up |
| `5d09563` | 2 layers, 16-dim, block=16 | Scaled back down — too slow token-by-token |

Output at `9857ea2`: loss was erratic (ranging from 0.03 to 4.0 — classic overfitting on a tiny model) and inference produced only ~4 tokens per sample:

```
step  200 / 1000 | loss 0.0332
step  600 / 1000 | loss 0.3019
step  800 / 1000 | loss 1.0971
step 1000 / 1000 | loss 0.2125

sample  1: Once upon a time
sample  8: Lily like a as a
```

The token-by-token loop was the bottleneck — couldn't scale the model up without training taking forever.

### Phase 2: Architecture overhaul

Rewrote the model based on [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c). This was the biggest single change — went from token-by-token to batched training with a modern transformer architecture.

| Commit | Config | What changed |
|---|---|---|
| `0e0e40e` | 5 layers, 128-dim, block=256, batch=32, 1000 steps | Batched forward, RoPE, flash attention, RMSNorm, weight tying, mixed precision, grad clipping, ReLU² |

New features in one commit: batched `(B, T)` training, `scaled_dot_product_attention`, RoPE, RMSNorm, weight tying, `GradScaler`, `clip_grad_norm_`. File renamed from `microgpt-colab.ipynb` → `microgpt-gpu-colab.ipynb` → `microgpt_fast.ipynb`.

Output at `0e0e40e`: loss dropped smoothly from 4.95 → 1.05, and inference produced much longer, more coherent samples (though still with some nonsense words and grammar):

```
step  100/1000 | loss 2.42
step  500/1000 | loss 1.32
step 1000/1000 | loss 1.05

sample 1: Once upon a time, there was ... The bird was not stink tight it was to shout. The
sample 2: One day, a little dog named Ben. Tim was every say and made a big time box. ...
```

### Phase 3: Training recipe tuning

With the architecture in place, iterated on training hyperparameters to push loss down.

| Commit | Config | What changed |
|---|---|---|
| `07f6b92` | steps 1000→3000, lr=5e-4 | More training steps |
| `73857d7` | batch 32→128, steps 3000→5000 | Bigger batches, more steps |
| `b88c021` | steps 5000→2000, added `torch.compile` | `torch.compile` made each step ~2× faster, so fewer steps needed |

### Phase 4: Training recipe improvements (free gains)

Changed only the training recipe — no model size change. Same compute, better loss.

| Commit | Config | What changed |
|---|---|---|
| `3937510` | ReLU→SiLU, linear LR→cosine+warmup, lr 5e-4→1e-3, steps→3000 | SiLU activation, cosine LR schedule, warmup, gradient clipping, higher learning rate |

Loss: ~0.99 → ~0.81. Words mostly real but grammar still broken.

### Phase 5: Model size increase (the breakthrough)

Loss plateaued at ~0.81 — a capacity ceiling. No training tricks could push past it.

| Commit | Config | What changed |
|---|---|---|
| `664b307` | n_embd 128→256, n_layer 5→6, batch 128→64, steps 3000→2000 | ~4× parameters per layer. Halved batch to fit GPU memory |

Loss: ~0.81 → ~0.77. Coherent sentences appeared, some nonsense words remained.

### Phase 6: Final tuning

Squeezed remaining quality from the training schedule.

| Commit | Config | What changed |
|---|---|---|
| `dede655` | steps 2000→3500, min_lr=1e-4, temperature 0.8→0.7 | min_lr floor prevents wasted steps at tail, more steps, lower sampling temperature |

Loss: ~0.77 → ~0.60. Output reads as simple children's stories.

### Key takeaways

- **Phase 2 (architecture overhaul) was the foundation** — batched training unlocked everything else.
- **Phase 4 (training recipe) was free** — same model, same compute, loss dropped 0.99 → 0.81.
- **Phase 5 (model size) was the breakthrough** — capacity ceiling at 0.81 could only be broken by a bigger model. Width (128→256) mattered more than depth.
- **Phase 6 (min_lr + more steps) was the final squeeze** — the cosine schedule was decaying LR to zero, wasting the last third of training.
- **Temperature at inference** (0.8→0.7) was the cheapest improvement — zero retraining, just pick higher-confidence tokens.

---

## Architecture

The model is a Llama-style decoder-only transformer. Each choice below replaces a simpler baseline from the original `microgpt.py`.

### Batched sequence processing

The original processes one token at a time. The fast version processes a full `(B, T)` tensor — 64 sequences × 256 tokens = 16,384 tokens per forward pass. This is the single biggest speedup and a prerequisite for everything else (flash attention, mixed precision, `torch.compile` all operate on batched tensors).

### Flash attention

```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Replaces an explicit Python loop over attention heads with a single fused CUDA kernel. Uses O(T) memory instead of O(T²) by never materialising the full attention matrix. On modded-nanogpt benchmarks this gave ~30% speedup (#12: 7.2 → 5.0 min).

### RoPE (Rotary Position Embeddings)

Instead of a learned position embedding table, RoPE encodes position by rotating query/key vectors:

```python
freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2) / head_dim))
```

Benefits: no extra parameters, better length generalisation, and the relative-position encoding means attention patterns transfer across sequence positions. ~30% faster in modded-nanogpt (#2: 45 → 31 min).

### Weight tying

The token embedding matrix `wte` doubles as the output projection (lm_head):

```python
# Embedding
x = F.embedding(tokens, state_dict['wte'])
# Output projection — same matrix
logits = F.linear(x, state_dict['wte'])
```

Saves `vocab_size × n_embd` parameters and gives the embedding matrix stronger gradient signal since it's updated from both the input and output sides.

### RMSNorm

```python
def rmsnorm(x):
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()
```

Simpler than LayerNorm — no mean subtraction, no bias, no learnable parameters. Slightly faster and equally effective for transformer training. Standard in Llama, Mistral, and all modern LLMs.

### SiLU activation

```python
x = F.silu(F.linear(x, mlp_fc1))  # SiLU = x * sigmoid(x)
```

Replaces ReLU in the MLP. SiLU (also called Swish) has smoother gradients around zero — ReLU has a hard cutoff that can create dead neurons. Standard in all Llama-family models.

### KV cache (inference only)

During generation, previously computed key/value tensors are cached per layer so each new token only requires a single forward pass instead of reprocessing the entire sequence. This makes generation O(T) instead of O(T²).

---

## Training

### Mixed precision (`float16` + GradScaler)

```python
with torch.amp.autocast('cuda', dtype=torch.float16):
    loss = F.cross_entropy(gpt_train(xb).view(-1, vocab_size), yb.view(-1))
scaler.scale(loss).backward()
```

Runs the forward pass in float16 (half the memory bandwidth), then `GradScaler` prevents underflow by scaling gradients up before the backward pass and back down before the optimizer step. ~2× throughput on T4.

### `torch.compile`

```python
gpt_train = torch.compile(gpt_train)
```

Traces the forward function and fuses operations into optimised CUDA kernels (e.g. combining matmul + activation into a single kernel launch). ~2× additional speedup. Only applied to the training forward pass — the inference `gpt()` function uses Python-level KV cache manipulation that doesn't benefit from compilation.

### Cosine LR with warmup and min_lr floor

```python
if step < warmup_steps:
    lr_t = learning_rate * step / warmup_steps
else:
    progress = (step - warmup_steps) / (num_steps - warmup_steps)
    lr_t = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

- **Warmup** (200 steps): prevents early training instability when parameters are still random.
- **Cosine decay**: smoothly reduces LR, spending more time at moderate learning rates than linear decay.
- **min_lr floor** (1e-4 = 10% of peak): prevents the tail of training from being wasted at near-zero LR. Without this, the last ~30% of steps contribute almost nothing.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(params, 1.0)
```

Caps the global gradient norm at 1.0. Prevents occasional large batches from causing loss spikes that destabilise training. Cheap insurance — almost no overhead.

### AdamW

```python
optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), eps=1e-10)
```

AdamW decouples weight decay from the gradient update (unlike Adam where weight decay is entangled with the adaptive learning rate). `beta2=0.95` (instead of the default 0.999) makes the second moment estimate more responsive, which helps with the non-stationary gradients typical in language model training.

---

## Model configuration

| Hyperparameter | Value | Role |
|---|---|---|
| `n_layer` | 6 | Number of transformer blocks |
| `n_embd` | 256 | Embedding / hidden dimension |
| `block_size` | 256 | Max context length (tokens) |
| `n_head` | 8 | Number of attention heads |
| `batch_size` | 64 | Sequences per gradient step |
| `num_steps` | 3500 | Training steps |
| `learning_rate` | 1e-3 | Peak learning rate |
| `min_lr` | 1e-4 | Learning rate floor (10% of peak) |
| `warmup_steps` | 200 | Linear warmup before cosine decay |
| `temperature` | 0.7 | Inference sampling temperature |

---

## Tuning hyperparameters

When tweaking the model, the training loss curve tells you what's wrong and what to change.

### Identifying the problem

| Symptom | Diagnosis | What's happening |
|---|---|---|
| Loss plateaus early and stays high (e.g. >1.0) | **Underfitting** | Model doesn't have enough capacity to learn the patterns in the data |
| Loss drops very low on training data but generated text is gibberish or repetitive | **Overfitting** | Model memorised the training data instead of learning general patterns |
| Loss drops quickly then flattens — more training steps don't help | **Capacity ceiling** | The model has learned everything it can at its current size |
| Loss is still dropping when training ends | **Undertrained** | The model needs more steps to converge |
| Loss spikes or diverges mid-training | **Training instability** | Learning rate too high, or missing gradient clipping |

### How to diagnose from the loss curve

- **Capacity ceiling vs undertrained:** If loss is flat for the last 30%+ of training, it's a capacity ceiling. If it's still visibly dropping at the end, you just need more steps.
- **Overfitting:** Compare loss on training data vs held-out data — a growing gap means overfitting. With small datasets, the model starts producing exact memorised phrases instead of novel text.
- **Underfitting:** Even after many steps, the loss stays well above what a larger model achieves on the same data.

### Typical solutions

| Problem | What to change | Example |
|---|---|---|
| **Capacity ceiling** | Increase `n_embd` (width) or `n_layer` (depth) | `n_embd`: 128 → 256 gave ~0.1 loss drop |
| **Undertrained** | Increase `num_steps` | 2000 → 3500 steps, check if loss is still falling |
| **Underfitting** | Larger model, higher learning rate, or better LR schedule | Switch linear decay → cosine with warmup |
| **Overfitting** | More training data, or reduce model size | 5000 → 20000 stories, or reduce `n_layer` |
| **Training instability** | Add gradient clipping, reduce learning rate, add warmup | `clip_grad_norm_(1.0)`, warmup 200 steps |
| **Slow convergence** | Increase `batch_size` (more stable gradients) or `learning_rate` | `batch_size`: 64 → 128 (if GPU memory allows) |

### Rules of thumb

- **Width (`n_embd`) is more impactful than depth (`n_layer`)** for small models. Doubling `n_embd` roughly 4× the parameter count per layer.
- **Bigger model + fewer steps often beats smaller model + more steps** within a fixed time budget.
- **Halve `batch_size` when you double model size** to stay within GPU memory.
- **Learning rate and model size are linked** — larger models generally need lower learning rates.
- **Temperature at inference** doesn't affect training but strongly affects output quality. Lower temperature (e.g. 0.7) reduces gibberish by picking higher-confidence tokens.

---

## Relative improvements (modded-nanogpt benchmarks)

These numbers are from the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), where each technique was measured in isolation on 8×H100 GPUs.

| Change | Relative improvement |
|---|---|
| Flash attention | ~30% faster (#12: 7.2 → 5.0 min) |
| RoPE | ~30% faster (#2: 45 → 31 min) |
| Weight tying | ~neutral speed, saves params (#8/#51) |
| RMSNorm | ~5–10% faster (standard in all records) |
| Mixed precision | ~5% faster (#10: 8.2 → 7.8 min) |
| `torch.compile` | ~8% faster (#7: 13.1 → 12.0 min) |
