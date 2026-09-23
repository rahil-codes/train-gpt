import argparse
import math
import os
import time
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class Config:
    vocab: int = 65
    block: int = 256
    layers: int = 6
    heads: int = 6
    emb: int = 384
    drop: float = 0.1
    bias: bool = False


class SelfAttn(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        assert c.emb % c.heads == 0
        self.heads = c.heads
        self.emb = c.emb
        self.drop = c.drop
        self.qkv = nn.Linear(c.emb, 3 * c.emb, bias=c.bias)
        self.proj = nn.Linear(c.emb, c.emb, bias=c.bias)
        self.resid_drop = nn.Dropout(c.drop)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(self.emb, dim=2)
        q, k, v = (t.view(B, T, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.drop if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.fc = nn.Linear(c.emb, 4 * c.emb, bias=c.bias)
        self.proj = nn.Linear(4 * c.emb, c.emb, bias=c.bias)
        self.drop = nn.Dropout(c.drop)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, c: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.emb)
        self.attn = SelfAttn(c)
        self.ln2 = nn.LayerNorm(c.emb)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok_emb = nn.Embedding(c.vocab, c.emb)
        self.pos_emb = nn.Embedding(c.block, c.emb)
        self.drop = nn.Dropout(c.drop)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.ln_f = nn.LayerNorm(c.emb)
        self.lm_head = nn.Linear(c.emb, c.vocab, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, T = idx.size()
        assert T <= self.c.block, "sequence longer than block"
        pos = torch.arange(T, dev=idx.dev)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss

    def make_opt(self, lr, wd, betas=(0.9, 0.95)):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "wd": wd},
            {"params": no_decay, "wd": 0.0},
        ]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, new_tokens, temp=1.0, top_k=None):
        self.eval()
        for _ in range(new_tokens):
            cur = idx[:, -self.c.block:]
            logits, _ = self(cur)
            logits = logits[:, -1, :] / max(temp, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def load_data(path):
    if path is None:
        path = "input.txt"
        if not os.path.exists(path):
            print(f"Downloading tiny-shakespeare to {path} ...")
            try:
                urllib.request.urlretrieve(SHAKESPEARE_URL, path)
            except Exception as e:
                raise SystemExit(
                    f"Could not download data ({e}). Pass a text file with --data yourfile.txt"
                )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TextData:
    """Char-level tokenizer + random-batch sampler over train/val splits."""

    def __init__(self, text, block, dev, val_frac=0.1):
        self.chars = sorted(set(text))
        self.vocab = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * (1 - val_frac))
        self.splits = {"train": data[:n], "val": data[n:]}
        self.block = block
        self.dev = dev

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def batch(self, split, bs):
        data = self.splits[split]
        ix = torch.randint(len(data) - self.block - 1, (bs,))
        x = torch.stack([data[i : i + self.block] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.block] for i in ix])
        if self.dev.type == "cuda":
            return x.pin_memory().to(self.dev, non_blocking=True), y.pin_memory().to(self.dev, non_blocking=True)
        return x.to(self.dev), y.to(self.dev)
def lr_step(it, args):
    """Linear warmup then cosine decay to min_lr."""
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    if it >= args.steps:
        return args.min_lr
    ratio = (it - args.warmup) / (args.steps - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)


@torch.no_grad()
def eval_loss(model, ds, args, ctx):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_steps)
        for k in range(args.eval_steps):
            x, y = ds.batch(split, args.bs)
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
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="use torch.compile (PyTorch 2+)")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.dev("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using dev: {dev}")
    use_cuda = dev.type == "cuda"
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
    text = load_data(args.data)
    ds = TextData(text, args.block, dev)
    print(f"Dataset: {len(text):,} chars, vocab size {ds.vocab}")
    c = Config(
        vocab=ds.vocab, block=args.block, layers=args.layers,
        heads=args.heads, emb=args.emb, drop=args.drop,
    )
    model = GPT(c).to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params / 1e6:.2f}M")
    opt = model.make_opt(args.lr, args.wd)
    train_model = torch.compile(model) if args.compile else model

    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    x, y = ds.batch("train", args.bs)
    for it in range(args.steps + 1):
        lr = lr_step(it, args)
        for g in opt.param_groups:
            g["lr"] = lr
        if it % args.eval_every == 0:
            losses = eval_loss(train_model, ds, args, ctx)
            print(f"[eval] step {it}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "config": c.__dict__,
                    "chars": ds.chars,
                    "iter": it,
                    "val_loss": best_val,
                }, os.path.join(args.out, "ckpt.pt"))
        if it == args.steps:
            break
        for step in range(args.accum):
            with ctx:
                _, loss = train_model(x, y)
                loss = loss / args.accum
            x, y = ds.batch("train", args.bs)
            scaler.scale(loss).backward()

        if args.clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if it % args.log_every == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {it:5d} | loss {loss.item() * args.accum:.4f} | lr {lr:.2e} | {dt * 1000:.0f} ms")
    print("\n--- Sample ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, dev=dev)
    out = model.generate(context, new_tokens=500, temp=0.8, top_k=40)
    print(ds.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
