#ruff: noqa 

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dlgrad import Tensor, nn


class GPTConfig:
    vocab_size = 0
    block_size = 128 # Context length
    n_layer = 4
    n_head = 4
    n_embd = 256
    dropout = 0.0
    learning_rate = 3e-4
    max_iters = 800
    batch_size = 16
    eval_interval = 10
    device = "cpu"

config = GPTConfig()

class CausalSelfAttention:
    def __init__(self, config):
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False, device=config.device)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False, device=config.device)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False, device=config.device)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False, device=config.device)

    def __call__(self, x, past_k=None, past_v=None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)  # (B, h, T, d)
        k = self.k_proj(x).reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).reshape((B, T, self.n_head, self.head_dim)).transpose(1, 2)

        if past_k is None and past_v is None:
            att = (q @ k.transpose(2, 3)) * self.scale  # (B, h, T, T)
            mask = Tensor.tril(Tensor.ones((T, T)), k=0.0)
            att = att.masked_fill(mask == Tensor(0.0), float('-inf'))
            att = att.softmax(dim=3)
            y = att @ v  # (B, h, T, d)
        else:
            k_np = np.concatenate((past_k.numpy(), k.numpy()), axis=2)
            v_np = np.concatenate((past_v.numpy(), v.numpy()), axis=2)
            k = Tensor(k_np, device=config.device)
            v = Tensor(v_np, device=config.device)
            att = (q @ k.transpose(2, 3)) * self.scale  # (B, h, T, T) but T=1 for q
            att = att.softmax(dim=3)
            y = att @ v

        y = y.transpose(1, 2).reshape((B, T, C))
        result = self.c_proj(y)
        return result, k, v  # Return updated KV for caching

class MLP:
    def __init__(self, config):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False, device=config.device)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False, device=config.device)

    def __call__(self, x):
        x = self.c_fc(x)
        x = x.relu()
        x = self.c_proj(x)
        return x

class Block:
    def __init__(self, config):
        self.ln1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x, past_k=None, past_v=None):
        norm1 = self.ln1(x)
        attn_out, new_k, new_v = self.attn(norm1, past_k, past_v)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_k, new_v

class GPT:
    def __init__(self, config):
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.wte(idx)
        pos_idxs = np.arange(T).astype(np.float32)
        pos_idxs_t = Tensor(pos_idxs, device=idx.device)
        pos_emb = self.wpe(pos_idxs_t)
        x = tok_emb + pos_emb
        if targets is not None:
            # Training: full forward, no cache
            for block in self.blocks:
                x, _, _ = block(x)  # Ignore KV
            x = self.ln_f(x)
            logits = self.lm_head(x)
            logits_flat = logits.reshape((B * T, self.config.vocab_size))
            targets_flat = targets.reshape((B * T, 1))
            loss = logits_flat.cross_entropy_loss(targets_flat)
            return logits, loss
        else:
            # Inference: full forward, return last logits and cache
            cache = []
            for block in self.blocks:
                x, k, v = block(x)
                cache.append((k, v))
            x = self.ln_f(x)
            logits_np = self.lm_head(x).numpy()[:, -1:, :]
            logits = Tensor(logits_np, device=idx.device)
            return logits, cache

    def forward_incremental(self, new_idx, pos_emb, cache):
        B, T = new_idx.shape
        assert T == 1, "Incremental forward expects single token"

        # Get token embedding
        tok_emb = self.wte(new_idx)  # Shape: (B, 1, n_embd)

        # Add position embedding - pos_emb should have shape (1, 1, n_embd) or (B, 1, n_embd)
        # If pos_emb has wrong shape, fix it:
        if len(pos_emb.shape) == 2:
            pos_emb = pos_emb.reshape((B, 1, -1))

        x = tok_emb + pos_emb

        new_cache = []
        for i, block in enumerate(self.blocks):
            past_k, past_v = cache[i]
            x, new_k, new_v = block(x, past_k, past_v)
            new_cache.append((new_k, new_v))

        x = self.ln_f(x)
        logits = self.lm_head(x)  # Shape: (B, 1, vocab_size)
        return logits, new_cache

    def _forward_incremental(self, new_idx, pos_emb, cache):
        B, T = new_idx.shape  # T=1
        assert T == 1
        tok_emb = self.wte(new_idx)
        x = tok_emb + pos_emb
        new_cache = []
        for i, block in enumerate(self.blocks):
            past_k, past_v = cache[i]
            x, new_k, new_v = block(x, past_k, past_v)
            new_cache.append((new_k, new_v))
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, 1, vocab)
        return logits, new_cache

    def generate(self, idx, max_new_tokens):
        idx_np = idx.numpy().astype(np.int32)
        cache = None
        current_len = idx_np.shape[1]
        for _ in range(max_new_tokens):
            print("---")
            s_total = time.perf_counter()
            if current_len >= self.config.block_size:
                # Crop sequence and reset cache
                idx_np = idx_np[:, -self.config.block_size :]
                cache = None  # Reset cache instead of trying to slide it
                current_len = self.config.block_size

            if cache is None:
                # Full forward (initial or after crop)
                s = time.perf_counter()
                idx_cond_np = idx_np
                idx_cond = Tensor(idx_cond_np.astype(np.float32), device=self.config.device)
                logits, cache = self(idx_cond)
                print(f"Full forward: {round((time.perf_counter() - s) * 1e3, 2)}ms")
            else:
                # Incremental forward
                s = time.perf_counter()
                new_idx_np = idx_np[:, -1:]
                new_idx = Tensor(new_idx_np.astype(np.float32), device=self.config.device)
                # Position should be the current length (index of the token we're generating)
                pos = current_len  # NOT current_len - 1
                # Shape should match input shape
                pos_t = Tensor(np.array([[pos]], dtype=np.float32), device=self.config.device)
                pos_emb = self.wpe(pos_t)
                logits, cache = self.forward_incremental(new_idx, pos_emb, cache)
                print(f"Incremental forward: {round((time.perf_counter() - s) * 1e3, 2)}ms")

            # Sample

            # Get the logits for the last token only
            logits_np = logits.numpy()

            # Handle shape - should be (batch, seq_len, vocab) or (batch, vocab)
            if len(logits_np.shape) == 3:
                logits_np = logits_np[0, -1, :]  # Get last token from first batch
            elif len(logits_np.shape) == 2:
                logits_np = logits_np[0, :]  # First batch
            else:
                logits_np = logits_np  # Already flat

            # TODO: Replace with dlgrad softmax after testing
            logits_np = logits_np - np.max(logits_np)
            exp_logits = np.exp(logits_np)
            probs_np = exp_logits / exp_logits.sum()

            # Sample next token
            idx_next = np.random.choice(self.config.vocab_size, p=probs_np)

            idx_np = np.concatenate([idx_np, [[idx_next]]], axis=1)
            current_len += 1
            print(f"Sample & append: {round((time.perf_counter() - s) * 1e3, 2)}ms")
            print(f"Total step: {round((time.perf_counter() - s_total) * 1e3, 2)}ms")
            del logits

        return Tensor(idx_np.astype(np.float32), device=self.config.device)

class CharTokenizer:
    """Character-level tokenizer - much smaller vocab!"""
    def __init__(self, text):
        # Get all unique characters
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        """Convert string to list of integers"""
        return [self.stoi[c] for c in s]

    def decode(self, l):
        """Convert list of integers to string"""
        return ''.join([self.itos[i] for i in l])

def clip_gradients(model, max_norm=1.0):
    # Calculate the total norm (length) of the gradient vector
    total_norm_sq = 0.0
    for p in nn.utils.get_parameters(model):
        if p.grad is not None:
            grad_val = p.grad.numpy()
            total_norm_sq += np.sum(grad_val ** 2)

    total_norm = np.sqrt(total_norm_sq)

    # Calculate scaling coefficient
    # If norm is small (e.g. 0.5), clip_coef > 1 (we don't scale up)
    # If norm is huge (e.g. 100.0), clip_coef < 1 (we shrink it)
    clip_coef = max_norm / (total_norm + 1e-6)

    # Scale gradients if they are exploding
    if clip_coef < 1.0:
        for p in nn.utils.get_parameters(model):
            if p.grad is not None:
                p.grad.data = p.grad.data * clip_coef

if __name__ == "__main__":
    if not os.path.exists('input.txt'):
        os.system('wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    with open('london-bridge-is-falling-down.txt') as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Characters: {''.join(tokenizer.chars)}")
    config.vocab_size = tokenizer.vocab_size

    model = GPT(config)

    total_params = 0
    for param in nn.utils.get_parameters(model):
        total_params += param.numel
    print(f"\nTotal parameters: {total_params/1e6:.2f}M")
    print("--")

    opt = nn.optim.Adam(params=nn.utils.get_parameters(model), lr=config.learning_rate)
    data = np.array(tokenizer.encode(text), dtype=np.int32)

    print("Starting training...")
    l = []

    pbar = tqdm(range(config.max_iters), desc="Training", unit="step")
    for i in pbar:
        ix = np.random.randint(0, len(data) - config.block_size, (config.batch_size,))
        x_batch = np.stack([data[j : j+config.block_size] for j in ix]).astype(np.float32)
        y_batch = np.stack([data[j+1 : j+config.block_size+1] for j in ix]).astype(np.float32)

        x_t = Tensor(x_batch, device=config.device)
        y_t = Tensor(y_batch, device=config.device)

        opt.zero_grad()
        logits, loss = model(x_t, y_t)
        loss_np = loss.numpy()
        l.append(loss_np)
        if np.isnan(loss_np):
            print(f"NaN detected in loss at step {i}! Breaking.")
            break

        loss.backward()

        # clip_gradients(model, max_norm=1.0)

        opt.step()

        loss_val = float(loss.numpy())
        pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        if i % config.eval_interval == 0:
            tqdm.write(f"Step {i}/{config.max_iters} | Loss: {loss_val:.4f}")

    print("\nTraining finished!")

    plt.plot(l)
    plt.title('Loss')
    plt.xlabel('iters')
    plt.ylabel('loss')
    plt.savefig('loss.png')

    print("\n=== Generating text ===")
    context = Tensor(np.array([[0]], dtype=np.float32), device=config.device)

    # Generate 500 tokens
    generated = model.generate(context, max_new_tokens=50)

    # Decode and print
    generated_tokens = generated.numpy()[0].astype(np.int32).tolist()
    print(tokenizer.decode(generated_tokens))


