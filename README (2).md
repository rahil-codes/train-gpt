# Minimal GPT in PyTorch

A compact, self-contained implementation of a GPT-style (decoder-only) Transformer with a full training loop. It trains a character-level language model on any plain-text file and can generate text from the result. Everything lives in a single file, `train_gpt.py`.

## Features

- **Model:** pre-LayerNorm Transformer blocks, causal multi-head self-attention (via PyTorch's fused `scaled_dot_product_attention`), GELU MLP, learned positional embeddings, tied input/output embeddings, GPT-2-style scaled initialization
- **Data:** character-level tokenizer built from your text; automatic 90/10 train/validation split
- **Training:**
  - AdamW with weight decay applied only to weight matrices
  - Linear warmup followed by cosine learning-rate decay
  - Gradient clipping and gradient accumulation
  - Mixed precision (bf16 when supported, otherwise fp16 with a `GradScaler`) on CUDA
  - Optional `torch.compile`
- **Checkpointing:** saves the best model (by validation loss) to `out/ckpt.pt`
- **Generation:** temperature and top-k sampling

## Requirements

- Python 3.8+
- PyTorch 2.0 or newer

```bash
pip install torch
```

## Quick start

```bash
python train_gpt.py
```

With no arguments, the script downloads the tiny-shakespeare dataset to `input.txt` (this requires internet access) and trains the default model. At the end of training it prints a generated sample.

To train on your own text:

```bash
python train_gpt.py --data my_text.txt
```

### Small model for CPU or a quick test

```bash
python train_gpt.py --n_layer 4 --n_head 4 --n_embd 128 \
                    --block_size 128 --batch_size 32 --max_iters 2000
```

## Command-line options

### Data and output

| Argument | Default | Description |
|---|---|---|
| `--data` | `None` | Path to a UTF-8 text file. If omitted, tiny-shakespeare is downloaded. |
| `--out_dir` | `out` | Directory for checkpoints. |
| `--seed` | `1337` | Random seed. |

### Model

| Argument | Default | Description |
|---|---|---|
| `--block_size` | `256` | Maximum context length. |
| `--n_layer` | `6` | Number of Transformer blocks. |
| `--n_head` | `6` | Attention heads per block (must divide `n_embd`). |
| `--n_embd` | `384` | Embedding / hidden dimension. |
| `--dropout` | `0.1` | Dropout probability. |

### Optimization

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | `64` | Sequences per micro-batch. |
| `--grad_accum` | `1` | Gradient accumulation steps (effective batch = `batch_size * grad_accum`). |
| `--max_iters` | `5000` | Total training iterations. |
| `--lr` | `3e-4` | Peak learning rate. |
| `--min_lr` | `3e-5` | Final learning rate after cosine decay. |
| `--warmup_iters` | `200` | Linear warmup steps. |
| `--weight_decay` | `0.1` | AdamW weight decay (matrices only). |
| `--grad_clip` | `1.0` | Max gradient norm; set to `0` to disable. |
| `--compile` | off | Enable `torch.compile`. |

### Logging and evaluation

| Argument | Default | Description |
|---|---|---|
| `--eval_interval` | `500` | Steps between train/val loss evaluations (and checkpoint saves). |
| `--eval_iters` | `50` | Batches averaged per evaluation. |
| `--log_interval` | `50` | Steps between training-loss log lines. |

## The default model

With the defaults (6 layers, 6 heads, 384-dim, 256-token context) the model has roughly 10M parameters. On a single modern GPU it trains in a few minutes on tiny-shakespeare. A validation loss around 1.5 is a reasonable target for that dataset. The exact figure depends on hardware, settings and seed.

## Output

- Progress is printed to the console: training loss, learning rate and step time every `--log_interval` steps, and train/val loss every `--eval_interval` steps.
- `out/ckpt.pt` holds the best checkpoint, with these keys:
  - `model`: the state dict
  - `config`: the model hyperparameters
  - `chars`: the character vocabulary
  - `iter`: the step the checkpoint was saved at
  - `val_loss`: the validation loss at that step

## Generating text from a saved checkpoint

The script only samples once, at the end of training. To generate later from a checkpoint:

```python
import torch
from train_gpt import GPT, GPTConfig

ckpt = torch.load("out/ckpt.pt", map_location="cpu")
model = GPT(GPTConfig(**ckpt["config"]))
model.load_state_dict(ckpt["model"])
model.eval()

chars = ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

prompt = "ROMEO:"
idx = torch.tensor([[stoi[c] for c in prompt if c in stoi]])
out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=40)
print("".join(itos[i] for i in out[0].tolist()))
```

## Code layout

| Component | Purpose |
|---|---|
| `GPTConfig` | Dataclass of model hyperparameters. |
| `CausalSelfAttention`, `MLP`, `Block` | Building blocks of the Transformer. |
| `GPT` | Full model: forward pass, optimizer setup, `generate()`. |
| `CharDataset` | Character tokenizer and random-batch sampler. |
| `get_lr` | Warmup + cosine learning-rate schedule. |
| `estimate_loss` | Averaged train/val loss evaluation. |
| `main` | Argument parsing and the training loop. |

## Extending it

- **Subword tokenization:** replace `CharDataset` with a BPE tokenizer such as `tiktoken`, and set `vocab_size` to match. The model code needs no changes.
- **Multi-GPU:** wrap the model in `DistributedDataParallel` and launch with `torchrun`.
- **Larger models:** increase `--n_layer`, `--n_head` and `--n_embd`, and use `--grad_accum` to reach a larger effective batch size without more memory.

## Notes

- Character-level models are ideal for learning and experimentation, but their output quality is limited compared with subword-tokenized models.
- The default download step needs network access. If it fails, pass a local file with `--data`.
- Checkpoints are only saved at evaluation steps, and only when validation loss improves.
