# microgpt_fast — Architecture and Training Choices

A detailed breakdown of every design decision in `microgpt_fast.ipynb` / `microgpt_fast.py`, why it was chosen, and how much it helps.

**Training time: ~3 min on a T4 GPU** (3500 steps, ~4M tokens seen during training, ~5M parameters). The model is character-level throughout — each token is a single character (vocab size 75).

---

## Contents

- [microgpt.py vs microgpt_fast.py](#microgptpy-vs-microgpt_fastpy) — side-by-side comparison of every significant change
- [Optimization journey](#optimization-journey) — how the model went from broken output to coherent stories, phase by phase
  - [Phase 1: PyTorch port](#phase-1-pytorch-port-token-by-token)
  - [Phase 2: Architecture overhaul](#phase-2-architecture-overhaul)
  - [Phase 3: Training recipe tuning](#phase-3-training-recipe-tuning)
  - [Phase 4: Training recipe improvements](#phase-4-training-recipe-improvements)
  - [Phase 5: Model size increase](#phase-5-model-size-increase-the-breakthrough)
  - [Phase 6: Final tuning](#phase-6-final-tuning)
  - [Phase 7: Inference vectorization](#phase-7-inference-vectorization)
  - [Key takeaways](#key-takeaways)
- [Architecture](#architecture) — what each component does and why it was chosen
- [Training](#training) — mixed precision, torch.compile, LR schedule, optimizer
- [Model configuration](#model-configuration) — final hyperparameters
- [Tuning hyperparameters](#tuning-hyperparameters) — how to diagnose and fix common problems
- [Relative improvements](#relative-improvements-modded-nanogpt-benchmarks) — benchmarks from modded-nanogpt
- [Glossary](#glossary) — plain-English definitions of all technical terms

---

## microgpt.py vs microgpt_fast.py

A side-by-side comparison of every significant change and why it was made.

| Aspect | `microgpt.py` | `microgpt_fast.py` | Why |
|---|---|---|---|
| **Compute** | Pure Python scalars, custom `Value` autograd | PyTorch tensors + CUDA | GPU parallelism |
| **Dataset** | `names.txt` (~32K names, ~5 chars each) | TinyStories (5000 stories, ~500 chars each) | Tests real language modelling — grammar, narrative, vocabulary |
| **Batching** | 1 document per step | 64 sequences × 256 tokens = 16,384 tokens per step | Utilises GPU memory bandwidth |
| **Compilation** | None | `torch.compile` | Kernel fusion, ~2× speedup |
| **Precision** | float64 (Python default) | float16 + GradScaler | Halves memory; faster matmuls on T4 |
| **Position encoding** | Learned `wpe` matrix | RoPE (Rotary) | No extra parameters; better length generalisation |
| **Attention** | Manual nested loops | `scaled_dot_product_attention` (Flash Attention) | Fused CUDA kernel; O(T) memory instead of O(T²) |
| **Activation** | ReLU | SiLU | Smoother gradients; standard in all Llama-family models |
| **Weight tying** | Separate `lm_head` matrix | `wte` reused as `lm_head` | Fewer parameters; stronger gradient signal |
| **Optimizer** | Manual Adam | PyTorch `AdamW` | Decoupled weight decay; less code |
| **LR schedule** | Linear decay to zero | Cosine with warmup + min_lr floor | Better convergence; min_lr prevents wasted steps at tail |
| **Gradient clipping** | None | `clip_grad_norm_(1.0)` | Prevents loss spikes |
| **KV cache** | Keys/values accumulate in lists | Same | Identical inference pattern |

The core **algorithm** is unchanged — `state_dict`, `for li in range(n_layer)`, attention + MLP blocks, causal masking, cross-entropy loss. Everything else is engineering to make it run ~1000× faster on a GPU.

---

## Optimization journey

The project started as a direct PyTorch port of the pure-Python `microgpt.py` — token-by-token training on a tiny model. Each phase below describes a problem that was hit, how it was identified, and what was done to fix it.

### Phase 1: PyTorch port (token-by-token)

The initial Colab notebook was a straight conversion of `microgpt.py` to PyTorch — same token-by-token training loop, same tiny model.

| Commit | Config | What changed |
|---|---|---|
| `9857ea2` | 1 layer, 16-dim, block=16, lr=0.01, 1000 steps | Initial port — ReLU, learned positions, separate lm_head |
| `3843e16` | 2 layers, 32-dim, block=32 | Tried scaling up |
| `5d09563` | 2 layers, 16-dim, block=16 | Scaled back down — too slow token-by-token |

**Problem:** At `9857ea2` loss was erratic (swinging between 0.03 and 4.0 — overfitting on a tiny model) and inference was capped at 16 characters per sample. This is a character-level model (`block_size = 16`), so the inference loop can produce at most 16 characters — most samples were just "Once upon a time".

**Attempted fix:** Scaling up at `3843e16` (`block_size = 32`) made things worse — training took over 3 minutes to reach step 300 (vs 1000 steps previously), and the generated text was longer but pure gibberish.

**Diagnosis:** The token-by-token loop was the bottleneck. Each training step processed a single character, so scaling the model up made training prohibitively slow without improving output quality.

### Phase 2: Architecture overhaul

**Fix:** Rewrote the model based on [EN10/modded-llama2.c](https://github.com/EN10/modded-llama2.c) — replaced the token-by-token loop with batched training and a modern transformer architecture.

| Commit | Config | What changed |
|---|---|---|
| `0e0e40e` | 5 layers, 128-dim, block=256, batch=32, 1000 steps | Batched forward, RoPE, flash attention, RMSNorm, weight tying, mixed precision, grad clipping, ReLU² |

**Result:** Loss dropped smoothly (4.95 → 1.32 → 1.05 over 1000 steps, ~2 min) and inference produced full sentences — though riddled with nonsense words (e.g. "repling", "gavore", "bottrowing", "lookoment").

**Remaining problem:** Loss was still above 1.0 — output had sentence structure but too many invented words and broken grammar.

### Phase 3: Training recipe tuning

**Diagnosis:** With loss still above 1.0, the model might just need more training — the loss curve was still dropping at step 1000.

| Commit | Config | What changed |
|---|---|---|
| `07f6b92` | steps 1000→3000, lr=5e-4 | More training steps |
| `73857d7` | batch 32→128, steps 3000→5000 | Bigger batches, more steps |
| `b88c021` | steps 5000→2000, added `torch.compile` | `torch.compile` made each step ~2× faster, so fewer steps needed |

**Result:** By `b88c021` loss reached ~0.99 in ~2 min. Generated text had mostly real words and sentence structure, but grammar was still broken and some nonsense words remained (e.g. "envelue", "destand", "tigerusting").

Between Phase 3 and 4, the notebook was renamed for clarity: `microgpt-colab.ipynb` → `microgpt-gpu-colab.ipynb` (`89caa81`) → `microgpt_fast.ipynb` (`63f4842`).

### Phase 4: Training recipe improvements

**Remaining problem:** Loss stuck around 0.99 — more steps weren't helping. The training recipe (linear LR decay, ReLU activation) might be leaving performance on the table.

**Fix:** Switched to better training defaults without changing the model size — same compute, better loss.

| Commit | Config | What changed |
|---|---|---|
| `3937510` | ReLU→SiLU, linear LR→cosine+warmup, lr 5e-4→1e-3, steps→3000 | SiLU activation, cosine LR schedule, warmup, higher learning rate |

**Result:** Loss: ~0.99 → ~0.81. Words mostly real but grammar still broken.

### Phase 5: Model size increase (the breakthrough)

**Remaining problem:** Loss plateaued at ~0.81 — no amount of training recipe tuning could push past it.

**Diagnosis:** Capacity ceiling. The 128-dim model had learned everything it could from the data. The loss curve was flat for the last 30%+ of training — a clear sign the model needed more parameters, not more steps.

**Fix:** Increased model width (`n_embd` 128→256) which roughly quadrupled the parameters per layer. Halved batch size to fit GPU memory.

| Commit | Config | What changed |
|---|---|---|
| `664b307` | n_embd 128→256, n_layer 5→6, batch 128→64, steps 3000→2000 | ~4× parameters per layer. Halved batch to fit GPU memory |

**Result:** Loss: ~0.81 → ~0.77. Coherent sentences appeared, some nonsense words remained.

### Phase 6: Final tuning

**Remaining problem:** Loss at 0.77 — output was coherent but still had occasional nonsense. The loss curve showed it was still dropping at step 2000, and the cosine LR schedule was decaying to zero, wasting the tail of training.

**Fix:** Added a `min_lr` floor (1e-4) so the last third of training stays productive, increased steps to 3500, and lowered inference temperature from 0.8 to 0.7 to pick higher-confidence tokens.

| Commit | Config | What changed |
|---|---|---|
| `dede655` | steps 2000→3500, min_lr=1e-4, temperature 0.8→0.7 | min_lr floor prevents wasted steps at tail, more steps, lower sampling temperature |

**Result:** Loss: ~0.77 → ~0.60. Output reads as simple children's stories:

> Once upon a time, there was a little boy named Tim. Tim had a magic cabin every day. He loved to play with his toy cars, even if he would go fast and play all day...

### Phase 7: Inference vectorization

**Problem:** After training was solid, inference was noticeably slow. Generating 5 samples of 200 tokens each took several seconds — disproportionate to how fast the trained model should be.

**Diagnosis:** The KV-cache attention loop was the bottleneck. For each new token, the code ran a Python `for h in range(n_head):` loop, stacking and computing attention per head one at a time:

```python
x_attn = []
for h in range(n_head):
    k_h = torch.stack([ki[h] for ki in keys[li]])
    v_h = torch.stack([vi[h] for vi in values[li]])
    x_attn.append(F.softmax(k_h @ q[h] / head_dim**0.5, dim=0) @ v_h)
x = F.linear(torch.cat(x_attn), attn_wo) + r
```

With 8 heads and 6 layers, this meant 48 Python loop iterations and 48 separate `torch.stack` calls per token — Python overhead dominated.

**Fix:** Replaced the loop with two `einsum` calls that compute all heads simultaneously:

```python
K = torch.stack(keys[li])    # (S, H, D)
V = torch.stack(values[li])  # (S, H, D)
attn = F.softmax(torch.einsum('hd,shd->sh', q, K) / head_dim**0.5, dim=0)
x = F.linear(torch.einsum('sh,shd->hd', attn, V).reshape(-1), attn_wo) + r
```

**Result:** Inference time for 5 × 200 token samples dropped from **39.7s → 7.1s** (~25 → ~140 tokens/s). The Python loop was the bottleneck — eliminating it lets the GPU handle all heads in a single kernel call.

### Key takeaways

- **Phase 2 (architecture overhaul) was the foundation** — batched training unlocked everything else.
- **Phase 4 (training recipe) was free** — same model, same compute, loss dropped 0.99 → 0.81.
- **Phase 5 (model size) was the breakthrough** — capacity ceiling at 0.81 could only be broken by a bigger model. Width (128→256) mattered more than depth.
- **Phase 6 (min_lr + more steps) was the final squeeze** — the cosine schedule was decaying LR to zero, wasting the last third of training.
- **Phase 7 (inference vectorization) was free** — same results, just eliminated a Python loop that was bottlenecking the GPU.
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

Attention over the cached K/V is computed across all heads simultaneously using `einsum`, replacing a Python `for h in range(n_head)` loop:

```python
K = torch.stack(keys[li])    # (S, H, D)
V = torch.stack(values[li])  # (S, H, D)
attn = F.softmax(torch.einsum('hd,shd->sh', q, K) / head_dim**0.5, dim=0)
x = F.linear(torch.einsum('sh,shd->hd', attn, V).reshape(-1), attn_wo)
```

This significantly speeds up inference — the per-head Python loop was the bottleneck during generation.

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
print(f"torch.compile cache: {os.getenv('TORCHINDUCTOR_CACHE_DIR', '~/.cache/torch/inductor')}")
```

Traces the forward function and fuses operations into optimised CUDA kernels (e.g. combining matmul + activation into a single kernel launch). ~2× additional speedup. Only applied to the training forward pass — the inference `gpt()` function uses Python-level KV cache manipulation that doesn't benefit from compilation.

**What the compiler needs to optimise well:** The loop structure inside the compiled function must be statically traceable. `for li in range(n_layer):` with a module-level constant lets the compiler fully unroll and specialise the loop. Replacing this with `for layer in layers:` — iterating a Python list of dicts — breaks static tracing, causing ~40s of JIT warmup at step 0 (vs ~3s normally) and measurably slower per-step execution.

**Cold vs warm compile:** The first run on a fresh Colab session compiles kernels and caches them in `/tmp/torchinductor_root`. The step 0 timestamp is the clearest indicator:

| Step 0 time | Meaning |
|---|---|
| ~3–4s | Cold compile — kernels being built for the first time |
| ~0.5–1s | Warm cache — compiled kernels reused from earlier in the session |

On a warm cache, training starts ~5× faster at step 0. The `torch.compile cache:` print confirms which path was taken.

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

### Vectorized batch sampling

```python
starts = torch.randint(0, len(all_tokens) - block_size - 1, (batch_size,), device=device)
idx = starts.unsqueeze(1) + torch.arange(block_size + 1, device=device)
tokens = all_tokens[idx]
```

Builds the entire batch with a single GPU index gather. Replaces a Python `for` loop + `torch.stack` that moved data the CPU per sample.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(params, 1.0)
```

Caps the global gradient norm at 1.0. Prevents occasional large batches from causing loss spikes that destabilise training. Cheap insurance — almost no overhead.

### AdamW

```python
optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.95), eps=1e-10,
                              fused=(device.type == 'cuda'))
```

`fused=True` on CUDA runs the entire optimizer step as a single kernel instead of separate kernel launches per parameter — a small free speedup.

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
| Step 0 takes ~40s instead of ~3s | **Compile regression** | A code change broke `torch.compile`'s ability to statically trace the loop — check that `for li in range(n_layer):` hasn't been replaced with a dynamic Python loop |

### Benchmarking training speed on Colab T4

Colab T4 instances share physical hardware — clock speeds, thermal state, and background load vary between allocations. This creates unavoidable noise in timing measurements:

- **~4% run-to-run variation is normal** and should be ignored.
- **>10–15% consistent difference** across multiple runs indicates a real regression.
- **Normalise by per-100-steps time**, not total time — different runs may have different `num_steps`.
- **Compare on the same session** (factory reset between runs) for a fair back-to-back comparison. Different Colab instances can differ by 15–20%.
- **Step 0 timing** is the most useful single number: it tells you whether `torch.compile` hit the cache (~0.6s) or compiled from scratch (~3.5s), and whether a code change broke compilation (~40s).

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

- **Width (`n_embd`) is more impactful than depth (`n_layer`)** for small models. Doubling `n_embd` roughly quadruples the parameter count per layer.
- **Bigger model + fewer steps often beats smaller model + more steps** within a fixed time budget.
- **Halve `batch_size` when you double model size** to stay within GPU memory.
- **Learning rate and model size are linked** — larger models generally need lower learning rates.
- **Temperature at inference** doesn't affect training but strongly affects output quality. Lower temperature (e.g. 0.7) reduces gibberish by picking higher-confidence tokens.

---

## Relative improvements (modded-nanogpt benchmarks)

These numbers are from the [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt), where each technique was measured in isolation on 8×H100 GPUs training a much larger model. They show the *relative* value of each technique — the absolute speedups differ on a T4 with a 5M-param model.

| Change | Relative improvement |
|---|---|
| Flash attention | ~30% faster (#12: 7.2 → 5.0 min) |
| RoPE | ~30% faster (#2: 45 → 31 min) |
| Weight tying | ~neutral speed, saves params (#8/#51) |
| RMSNorm | ~5–10% faster (standard in all records) |
| Mixed precision | ~5% faster (#10: 8.2 → 7.8 min) |
| `torch.compile` | ~8% faster (#7: 13.1 → 12.0 min) |

---

## Glossary

Plain-English explanations of the technical terms used in this document.

### Core concepts

| Term | What it means |
|---|---|
| **Token** | The smallest unit the model reads and writes. In this project, each token is a single character (letter, space, punctuation). Other models use words or word-pieces. |
| **Vocab / vocab size** | The complete set of tokens the model knows — here, 75 characters (a–z, A–Z, digits, punctuation, plus a special start-of-story marker). |
| **Loss** | A number that measures how wrong the model's predictions are. Lower = better. A loss of 0 would mean perfect prediction; in practice it never reaches 0. |
| **Training step** | One round of "show the model some data, measure how wrong it is, adjust its weights". The model gets a little better each step. |
| **Inference** | Using the trained model to generate new text (as opposed to training it). |
| **Parameters** | The numbers inside the model that get adjusted during training. More parameters = the model can learn more complex patterns, but needs more memory and time. Think of them as the model's "brain cells". |
| **Overfitting** | When the model memorises the training data instead of learning general patterns. Like studying only past exam answers — you ace those exact questions but fail on new ones. |
| **Underfitting** | The opposite — the model hasn't learned enough, even from the training data. Like not studying enough. |
| **Capacity ceiling** | When a model is too small to learn anything more from the data, no matter how long you train it. The only fix is a bigger model. |

### Model architecture

| Term | What it means |
|---|---|
| **Transformer** | The type of neural network architecture used by GPT, Llama, and nearly all modern language models. It uses "attention" to let each token look at all previous tokens when making predictions. |
| **Decoder-only** | A transformer that only generates text (as opposed to encoder-decoder models that also read/summarise input). All GPT-style models are decoder-only. |
| **Layer (`n_layer`)** | One complete processing block in the model. Data flows through layers sequentially — more layers = the model can learn more abstract patterns. Like adding more floors to a building. |
| **Embedding dimension (`n_embd`)** | The size of the internal representation for each token — how many numbers are used to describe it. A 256-dim embedding means each token is represented by a list of 256 numbers. Bigger = richer representation. |
| **Attention head (`n_head`)** | Attention is split into multiple "heads" that each focus on different patterns (e.g. one might track grammar, another might track meaning). More heads = more diverse pattern detection. |
| **Block size / context length** | The maximum number of tokens the model can "see" at once. With `block_size = 256`, the model can look back at the last 256 characters when predicting the next one. |
| **RoPE (Rotary Position Embeddings)** | A way to tell the model where each token is in the sequence by rotating its numbers. The alternative (learned positions) requires storing a separate table, and doesn't generalise to lengths not seen during training. |
| **RMSNorm** | A way to keep numbers inside the model from getting too large or too small between layers. Simpler and faster than the older "LayerNorm" method. |
| **Weight tying** | Reusing the same table of numbers for both looking up tokens (input) and predicting tokens (output). Saves memory and actually improves learning. |
| **Flash attention** | A faster way to compute attention that avoids creating a huge intermediate matrix. Same result, less memory, much faster. |
| **KV cache** | During text generation, the model saves its intermediate work (keys and values) so it doesn't have to redo calculations for tokens it's already processed. |
| **MLP (Multi-Layer Perceptron)** | The "feed-forward" part of each transformer layer — a simple two-layer neural network that processes each token independently. |
| **Activation function (ReLU, SiLU)** | A function applied between layers that introduces non-linearity — without it, stacking layers would be no better than one layer. ReLU zeroes out negative values; SiLU is smoother and generally works better. |
| **lm_head** | The final layer that converts the model's internal representation into a prediction over the vocabulary — "which token comes next?" |

### Training

| Term | What it means |
|---|---|
| **Batch / batch size** | How many sequences the model processes at once before updating its weights. Bigger batches = more stable learning, but need more GPU memory. |
| **Learning rate (LR)** | How big a step the model takes when adjusting its weights each training step. Too high = unstable (loss jumps around). Too low = learns very slowly. |
| **Cosine LR schedule** | A way to gradually reduce the learning rate over training that follows a cosine curve — starts high, slowly decreases, flattens at the end. Better than a straight line decrease. |
| **Warmup** | Starting with a very small learning rate and gradually increasing it over the first few hundred steps. Prevents the model from making wild adjustments when its weights are still random. |
| **min_lr floor** | A minimum learning rate that the schedule never drops below. Without it, the LR decays to zero and the model stops learning in the final portion of training. |
| **Gradient** | The direction and amount to adjust each parameter to reduce the loss. Calculated from the training data each step. |
| **Gradient clipping** | Capping the size of gradients so no single step makes a huge change to the model. Prevents sudden loss spikes. |
| **AdamW** | The optimiser — the algorithm that decides exactly how to update the model's parameters using the gradients. AdamW is the standard choice for training transformers. |
| **Mixed precision (float16)** | Using smaller numbers (16-bit instead of 32-bit) during training to run ~2× faster. A `GradScaler` prevents the smaller numbers from causing rounding errors. |
| **`torch.compile`** | A PyTorch feature that analyses your model and automatically fuses operations into faster GPU code. Like a compiler for neural networks. |
| **Forward pass** | Running data through the model to get predictions. |
| **Backward pass** | Calculating gradients by running backward through the model — figuring out how to adjust each parameter. |
| **Cross-entropy loss** | The specific loss function used — measures how far the model's predicted probability distribution is from the actual next character. |

### Inference

| Term | What it means |
|---|---|
| **Temperature** | Controls randomness when generating text. Lower temperature (e.g. 0.7) = the model picks higher-confidence tokens, producing more predictable text. Higher temperature = more creative but more likely to produce nonsense. Temperature of 1.0 = no adjustment. |
| **Sampling** | Randomly picking the next token based on the model's predicted probabilities, rather than always picking the most likely one. |
| **Logits** | The raw scores the model outputs for each possible next token, before converting to probabilities. |

### Hardware & tools

| Term | What it means |
|---|---|
| **GPU (Graphics Processing Unit)** | A chip designed for parallel computation — much faster than a CPU for the matrix maths that neural networks rely on. |
| **T4** | An NVIDIA GPU commonly available for free on Google Colab. Not the fastest, but enough to train small models. |
| **CUDA** | NVIDIA's software platform that lets programs run computations on their GPUs. |
| **PyTorch** | The Python library used to build and train the model. Handles all the GPU acceleration, automatic gradient calculation, etc. |
| **Google Colab** | A free online notebook environment from Google that provides GPU access — where this model was developed and trained. |
| **Kernel (CUDA kernel)** | A small program that runs on the GPU. Flash attention, for example, replaces many small GPU programs with one efficient one. |
