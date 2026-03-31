# How `microgpt.py` works

A step-by-step walkthrough of the pure-Python GPT — no PyTorch, no NumPy, just `os`, `math`, and `random`.

> *"The most atomic way to train and run inference for a GPT in pure, dependency-free Python. This file is the complete algorithm. Everything else is just efficiency."* — @karpathy

---

## 1. Dataset

```python
docs = [line.strip() for line in open('input.txt') if line.strip()]
```

The dataset is ~32,000 first names from [karpathy/makemore](https://github.com/karpathy/makemore), downloaded automatically on first run. Each name is a single "document". The list is shuffled with a fixed seed for reproducibility.

---

## 2. Tokenizer

```python
uchars = sorted(set(''.join(docs)))  # unique characters → token ids 0..n-1
BOS = len(uchars)                    # special Beginning of Sequence token
vocab_size = len(uchars) + 1         # +1 for BOS
```

A character-level tokenizer built dynamically from the dataset. Every unique character gets an integer id, and one extra id is reserved for a special **BOS** (Beginning of Sequence) token used to mark the start and end of each name.

---

## 3. Autograd — the `Value` class

The heart of the file. Every scalar in the model is wrapped in a `Value` object that tracks:

- **`data`** — the scalar's current value (forward pass)
- **`grad`** — the derivative of the loss with respect to this scalar (backward pass)
- **`_children`** — parent nodes in the computation graph
- **`_local_grads`** — the local derivative of this operation with respect to each child

### Supported operations

| Operation | Local gradients |
|---|---|
| `a + b` | $\frac{\partial}{\partial a} = 1$, $\frac{\partial}{\partial b} = 1$ |
| `a * b` | $\frac{\partial}{\partial a} = b$, $\frac{\partial}{\partial b} = a$ |
| `a ** n` | $\frac{\partial}{\partial a} = n \cdot a^{n-1}$ |
| `log(a)` | $\frac{\partial}{\partial a} = \frac{1}{a}$ |
| `exp(a)` | $\frac{\partial}{\partial a} = e^a$ |
| `relu(a)` | $\frac{\partial}{\partial a} = \begin{cases} 1 & a > 0 \\ 0 & \text{otherwise} \end{cases}$ |

Compound operations (`-`, `/`, etc.) are built from these primitives via operator overloading.

### Backpropagation

```python
def backward(self):
    # 1. Topological sort of the computation graph
    # 2. Walk nodes in reverse order
    # 3. Accumulate: child.grad += local_grad * parent.grad
```

`backward()` performs a topological sort of the graph, then propagates gradients from the loss back to every parameter using the chain rule. This is the same algorithm that PyTorch's `autograd` implements — just on individual scalars instead of tensors.

---

## 4. Model parameters

```python
n_layer    = 1    # transformer layers
n_embd     = 16   # embedding dimension
block_size = 16   # max context length
n_head     = 4    # attention heads
head_dim   = 4    # n_embd // n_head
```

All weights are stored in a flat `state_dict` dictionary of 2D lists of `Value` objects:

| Weight | Shape | Purpose |
|---|---|---|
| `wte` | `(vocab_size, n_embd)` | Token embeddings |
| `wpe` | `(block_size, n_embd)` | Learned position embeddings |
| `lm_head` | `(vocab_size, n_embd)` | Output projection (tokens → logits) |
| `attn_wq/wk/wv/wo` | `(n_embd, n_embd)` | Attention query, key, value, output projections |
| `mlp_fc1` | `(4*n_embd, n_embd)` | MLP expansion layer |
| `mlp_fc2` | `(n_embd, 4*n_embd)` | MLP contraction layer |

Weights are initialized from a Gaussian distribution with standard deviation 0.08.

---

## 5. Architecture — the `gpt()` function

The model follows GPT-2 with minor simplifications: **RMSNorm** instead of LayerNorm, **ReLU** instead of GeLU, and no biases.

### Forward pass for a single token

```
token_id, pos_id
       │
   ┌───┴───┐
   │ embed  │  wte[token_id] + wpe[pos_id]
   └───┬───┘
       │
   ┌───┴───┐
   │rmsnorm │
   └───┬───┘
       │
  ┌────┴────┐
  │ layer 0 │
  └────┬────┘
       │
   ┌───┴───┐
   │lm_head│  → logits (one score per vocab token)
   └───────┘
```

Each transformer layer has two sub-blocks with residual connections:

### 5a. Multi-head attention

1. **Normalize** the input with RMSNorm
2. **Project** into queries (Q), keys (K), and values (V) via learned linear layers
3. **Append** K and V to a running cache (enabling autoregressive generation)
4. **Split** Q, K, V into `n_head` heads of dimension `head_dim`
5. **Score** each head: $\text{attn} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)$
6. **Aggregate**: weighted sum of values by attention weights
7. **Concatenate** heads and project back with `attn_wo`
8. **Add** the residual connection

### 5b. MLP (feed-forward)

1. **Normalize** with RMSNorm
2. **Expand** to 4× the embedding dimension (`mlp_fc1`)
3. **ReLU** activation
4. **Contract** back to the embedding dimension (`mlp_fc2`)
5. **Add** the residual connection

### Helper functions

- **`linear(x, w)`** — matrix-vector multiply (dot product of each row of `w` with `x`)
- **`softmax(logits)`** — numerically stable softmax (subtracts max before exp)
- **`rmsnorm(x)`** — root-mean-square normalization: $x_i \cdot \left(\frac{1}{n}\sum x_i^2 + \epsilon\right)^{-0.5}$

---

## 6. Training loop

```python
for step in range(1000):
    # 1. Pick a name, tokenize: [BOS, c1, c2, ..., cn, BOS]
    # 2. Forward each token through gpt(), get logits
    # 3. Cross-entropy loss: -log(prob of correct next token)
    # 4. Backward pass: loss.backward()
    # 5. Adam optimizer update
```

### Training details

- **One document per step** — no batching (pure sequential)
- **Loss** — average cross-entropy over all positions in the sequence
- **Optimizer** — Adam with linear learning rate decay from 0.01 to 0
- **Hyperparameters** — $\beta_1 = 0.85$, $\beta_2 = 0.99$, $\epsilon = 10^{-8}$

After the forward pass builds the full computation graph (every `Value` node connected by operations), `loss.backward()` traverses it in reverse to compute gradients for all parameters simultaneously.

---

## 7. Inference

```python
temperature = 0.5
for sample_idx in range(20):
    token_id = BOS
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
```

Generation is autoregressive: feed BOS, sample the next token from the probability distribution, feed that token back in, repeat until BOS is emitted again (end of name) or the context window is full.

**Temperature** controls randomness — lower values (closer to 0) make the model more confident/repetitive, higher values (closer to 1) increase diversity.

The key/value cache (`keys`, `values`) accumulates K and V vectors from previous positions so each step only computes attention for the new token rather than reprocessing the entire sequence.

---

## Why it's slow (and that's the point)

Every number is a `Value` object. A single forward pass through one name builds a graph of thousands of `Value` nodes, and backprop walks all of them. There are no vectorized operations — every multiply and add is a Python function call.

This makes it **easy to read and understand** but impractical for real workloads. The PyTorch versions (`microgpt_torch.py`, `microgpt_fast.py`) replace `Value` with tensors and GPU kernels for orders-of-magnitude speedups while keeping the same algorithm.
