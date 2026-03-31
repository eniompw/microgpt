import os
import random
import json
import time
import urllib.request

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Reproducibility
random.seed(42)
torch.manual_seed(42)

# Device setup — will use T4 GPU on Colab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Dataset Download and Preparation ─────────────────────────────────────────
if not os.path.exists('input.txt'):
    print("Downloading TinyStories dataset from HuggingFace...")
    stories = []
    base_url = 'https://datasets-server.huggingface.co/rows?dataset=karpathy/tinystories-gpt4-clean&config=default&split=train'
    for offset in range(20000, 21000, 100):  # rows 20000+ are the train split
        url = f'{base_url}&offset={offset}&limit=100'
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
        for item in data['rows']:
            stories.append(item['row']['text'])
        print(f"  fetched {len(stories)} stories...", end='\r')
    print()
    with open('input.txt', 'w') as f:
        for story in stories:
            f.write(json.dumps(story) + '\n')
    print("Saved to input.txt")
else:
    print("input.txt already exists, skipping download.")

docs = [json.loads(line) for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
print(f"sample doc: {docs[0][:120]}...")

# ── Tokenizer Setup ───────────────────────────────────────────────────────────
# Character-level vocabulary — all 74 chars present in the dataset
uchars = sorted('\n !"$\',-.' + '0123456789:;?' + 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 'abcdefghijklmnopqrstuvwxyz')
BOS = len(uchars)          # special Beginning-of-Sequence token id
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")
print(f"BOS token id: {BOS}")

encode = lambda s: [uchars.index(ch) for ch in s]
decode = lambda ids: ''.join(uchars[i] for i in ids)

# ── Hyperparameters ───────────────────────────────────────────────────────────
n_layer    = 5       # transformer depth
n_embd     = 128     # embedding dim
block_size = 256     # context window
n_head     = 8       # attention heads
head_dim   = n_embd // n_head
batch_size = 128     # sequences per gradient step

# ── Weight Initialization ─────────────────────────────────────────────────────
matrix = lambda nout, nin: torch.randn(nout, nin, device=device) * 0.02

state_dict = {
    'wte': matrix(vocab_size, n_embd),   # token embeddings (weight-tied to lm_head)
}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

params = list(state_dict.values())
for p in params:
    p.requires_grad_(True)

total_params = sum(p.numel() for p in params)
print(f"num params: {total_params:,}")
print(f"tokens per iter: {batch_size * block_size:,}")

# ── Model Architecture ────────────────────────────────────────────────────────
def rmsnorm(x):
    """RMSNorm along last dim — works for any shape."""
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()

# RoPE (Rotary Position Embeddings)
freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
freqs = torch.outer(torch.arange(block_size, device=device).float(), freqs)
rope_cos, rope_sin = freqs.cos(), freqs.sin()   # (block_size, head_dim//2)

def apply_rope(x, cos, sin):
    """x: (B,T,H,D) or (H,D). cos/sin: (T,D//2) or (D//2,)"""
    d = x.dtype
    x = x.float().unflatten(-1, (-1, 2))
    x_r, x_i = x[..., 0], x[..., 1]
    if x.dim() == 5:  # batched
        cos = cos.view(1, -1, 1, cos.shape[-1])
        sin = sin.view(1, -1, 1, sin.shape[-1])
    return torch.stack([x_r*cos - x_i*sin, x_r*sin + x_i*cos], -1).flatten(-2).to(d)

def gpt_train(tokens):
    """tokens: (B, T) long -> logits: (B, T, padded_vocab)"""
    bsz, seqlen = tokens.shape
    x = rmsnorm(F.embedding(tokens, state_dict['wte']))
    cos, sin = rope_cos[:seqlen], rope_sin[:seqlen]
    for li in range(n_layer):
        r = x
        x = rmsnorm(x)
        q = F.linear(x, state_dict[f'layer{li}.attn_wq']).view(bsz, seqlen, n_head, head_dim)
        k = F.linear(x, state_dict[f'layer{li}.attn_wk']).view(bsz, seqlen, n_head, head_dim)
        v = F.linear(x, state_dict[f'layer{li}.attn_wv']).view(bsz, seqlen, n_head, head_dim)
        q, k = apply_rope(q, cos, sin), applyvocab_size)"""
    bsz, seqlen = tokens.shape
    x = rmsnorm(F.embedding(tokens, state_dict['wte']))
    cos, sin = rope_cos[:seqlen], rope_sin[:seqlen]
    for li in range(n_layer):
        r = x
        x = rmsnorm(x)
        q = F.linear(x, state_dict[f'layer{li}.attn_wq']).view(bsz, seqlen, n_head, head_dim)
        k = F.linear(x, state_dict[f'layer{li}.attn_wk']).view(bsz, seqlen, n_head, head_dim)
        v = F.linear(x, state_dict[f'layer{li}.attn_wv']).view(bsz, seqlen, n_head, head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
        ).transpose(1,2).contiguous().view(bsz, seqlen, -1)
        x = F.linear(x, state_dict[f'layer{li}.attn_wo']) + r
        r = x
        x = rmsnorm(x)
        x = F.relu(F.linear(x, state_dict[f'layer{li}.mlp_fc1']))
        x = F.linear(x, state_dict[f'layer{li}.mlp_fc2']) + r
    return F.linear(rmsnorm(x), state_dict['wte'])   # weight-tied lm_head

gpt_train = torch.compile(gpt_train)  # fuse GPU kernels for ~2x speedupd_dim)
        v = F.linear(x, state_dict[f'layer{li}.attn_wv']).view(n_head, head_dim)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            k_h = torch.stack([ki[h] for ki in keys[li]])
            v_h = torch.stack([vi[h] for vi in values[li]])
            x_attn.append(F.softmax(k_h @ q[h] / head_dim**0.5, dim=0) @ v_h)
        x = F.linear(torch.cat(x_attn), state_dict[f'layer{li}.attn_wo']) + r
        r = x
        x = rmsnorm(x)
        x = F.relu(F.linear(x, state_dict[f'layer{li}.mlp_fc1'])).square()
        x = F.linear(x, state_dict[f'layer{li}.mlp_fc2']) + r
    return F.linear(rmsnorm(x), state_dict['wte'])

# ── Training Loop ─────────────────────────────────────────────────────────────
all_tokens = []
for doc in docs:
    all_tokens.extend([BOS] + encode(doc))
all_tokens.append(BOS)
all_tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
print(f"Total tokens: {len(all_tokens):,}")

num_steps     = 1000
base_lr       = 5e-4
min_lr        = base_lr / 10
cooldown_frac = 0.1
stable_steps  = int(num_steps * (1 - cooldown_frac))

embed_params  = [state_dict['wte']]
matrix_params = [p for n, p in state_dict.items() if n != 'wte']
optimizer = torc2000
learning_rate = 5e-4

optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.95), eps=1e-10)

# Mixed precision (float16 on T4)        g['lr'] = g['initial_lr'] * lr_mult(step)

    if step % 100 == 0:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            ix = torch.randint(0, len(all_tokens) - block_size - 1, (batch_size,))
            xb = torch.stack([all_tokens[i:i+block_size] for i in ix])
            yb = torch.stack([all_tokens[i+1:i+block_size+1] for i in ix])
            el = F.cross_entropy(gpt_train(xb).view(-1, padded_vocab), yb.view(-1)).item()
        print(f"step {step:4d}/{num_steps} | loss {el:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")

    if step >= num_steps:
        break

    optimizer.zero_grad()
    for _ in range(grad_accum_steps):
        ix = torch.randint(0, len(all_tokens) - block_size - 1, (batch_size,))
        xb = torch.stack([all_tokens[i:i+block_size] for i in ix])
        yb = torch.stack([all_tokens[i+1:i+block_size+1] for i in ix])
        with torch.amp.autocast('cuda', dtype=torch.float16):
            loss = F.cross_entropy(gpt_train(xb).view(-1, padded_vocab), yb.view(-1)) / grad_accum_steps
        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    # Linear learning rate decay
    lr_t = learning_rate * (1 - step / num_steps)
    for g in optimizer.param_groups:
        g['lr'] = lr_t

    if step % 100 == 0:
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            ix = torch.randint(0, len(all_tokens) - block_size - 1, (batch_size,))
            xb = torch.stack([all_tokens[i:i+block_size] for i in ix])
            yb = torch.stack([all_tokens[i+1:i+block_size+1] for i in ix])
            el = F.cross_entropy(gpt_train(xb).view(-1, vocab_size), yb.view(-1)).item()
        print(f"step {step:4d}/{num_steps} | loss {el:.4f} | lr {lr_t:.2e} | {time.time()-t0:.1f}s")

    if step >= num_steps:
        break

    optimizer.zero_grad()
    ix = torch.randint(0, len(all_tokens) - block_size - 1, (batch_size,))
    xb = torch.stack([all_tokens[i:i+block_size] for i in ix])
    yb = torch.stack([all_tokens[i+1:i+block_size+1] for i in ix])
    with torch.amp.autocast('cuda', dtype=torch.float16):
        loss = F.cross_entropy(gpt_train(xb).view(-1, vocab_size), yb.view(-1))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    loss_history.append(loss.item()
                break
            sample.append(uchars[token_id])
    print(f"sample {sample_idx+1}:\n{''.join(sample)}\n")
