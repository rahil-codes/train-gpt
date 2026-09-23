"""
Minimal GPT-style (decoder-only) Transformer in PyTorch.

Character-level language model, trained on any plain-text file.

Usage:
    python train_gpt.py --data input.txt
    python train_gpt.py                      # downloads tiny-shakespeare if no file is given

Requirements: torch >= 2.0
"""

import argparse
import math
import os
import time
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256      # max context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        # Fused, memory-efficient attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual projections
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas=(0.9, 0.95)):
        # Apply weight decay only to matrices (not biases / LayerNorm / embeddings' 1D params)
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_text(path):
    if path is None:
        path = "input.txt"
        if not os.path.exists(path):
            print(f"Downloading tiny-shakespeare to {path} ...")
            try:
                urllib.request.urlretrieve(SHAKESPEARE_URL, path)
            except Exception as e:
                raise SystemExit(
                    f"Could not download dataset ({e}). Pass a text file with --data yourfile.txt"
                )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharDataset:
    """Char-level tokenizer + random-batch sampler over train/val splits."""

    def __init__(self, text, block_size, device, val_frac=0.1):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.splits = {"train": data[:n], "val": data[n:]}
        self.block_size = block_size
        self.device = device

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, split, batch_size):
        data = self.splits[split]
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block_size] for i in ix])
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def get_lr(it, args):
    """Linear warmup then cosine decay to min_lr."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    if it >= args.max_iters:
        return args.min_lr
    ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)


@torch.no_grad()
def estimate_loss(model, ds, args, ctx):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = ds.get_batch(split, args.batch_size)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="path to a plain-text file")
    p.add_argument("--out_dir", type=str, default="out")
    # model
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimization
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging / eval
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="use torch.compile (PyTorch 2+)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mixed precision: bf16 if supported, else fp16 + GradScaler, else fp32
    use_cuda = device.type == "cuda"
    if use_cuda and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif use_cuda:
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Data
    text = load_text(args.data)
    ds = CharDataset(text, args.block_size, device)
    print(f"Dataset: {len(text):,} chars, vocab size {ds.vocab_size}")

    # Model
    cfg = GPTConfig(
        vocab_size=ds.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout,
    )
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.2f}M")
    optimizer = model.configure_optimizer(args.lr, args.weight_decay)
    train_model = torch.compile(model) if args.compile else model

    os.makedirs(args.out_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    x, y = ds.get_batch("train", args.batch_size)
    for it in range(args.max_iters + 1):
        # LR schedule
        lr = get_lr(it, args)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Evaluation + checkpointing
        if it % args.eval_interval == 0:
            losses = estimate_loss(train_model, ds, args, ctx)
            print(f"[eval] step {it}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "chars": ds.chars,
                    "iter": it,
                    "val_loss": best_val,
                }, os.path.join(args.out_dir, "ckpt.pt"))
        if it == args.max_iters:
            break

        # Training step (with gradient accumulation)
        for micro in range(args.grad_accum):
            with ctx:
                _, loss = train_model(x, y)
                loss = loss / args.grad_accum
            x, y = ds.get_batch("train", args.batch_size)  # prefetch next batch
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if it % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {it:5d} | loss {loss.item() * args.grad_accum:.4f} | lr {lr:.2e} | {dt * 1000:.0f} ms")

    # Sample from the trained model
    print("\n--- Sample ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    print(ds.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
