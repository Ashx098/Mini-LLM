# Training Module

This module contains all training infrastructure including data loading, optimization, learning rate scheduling, and the main training loop.

## Overview

The training system implements modern best practices for large-scale language model training:
- **Memory-mapped data loading** for efficient I/O
- **Gradient accumulation** for large effective batch sizes
- **Mixed precision training** (FP16/BF16)
- **Cosine learning rate schedule** with warmup
- **AdamW optimizer** with decoupled weight decay
- **Gradient clipping** for stability
- **Checkpointing** for resumable training

## Directory Structure

```
train/
├── dataloader.py        # Memory-mapped data loading
├── optimizer.py         # AdamW with weight decay grouping
├── lr_scheduler.py      # Cosine schedule with warmup
├── train.py            # Main training loop
├── config.yaml         # Production training config
└── config_test.yaml    # Test/debug config
```

## Components

### 1. DataLoader (`dataloader.py`)

Efficiently loads tokenized data using memory mapping.

```python
class DataLoader:
    def __init__(self, data_dir, batch_size, block_size, split="train"):
        # Memory-map the binary file (no RAM loading!)
        filename = f"{data_dir}/{split}.bin"
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    def get_batch(self, device="cpu"):
        # Random sampling
        ix = torch.randint(0, len(self.data) - block_size, (batch_size,))
        
        # Stack batch with x/y shift
        x = torch.stack([self.data[i:i+block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+1+block_size] for i in ix])
        
        return x.to(device), y.to(device)
```

**Example: Next-Token Prediction**

```
Data: [1, 334, 3855, 288, 267, 2959, 354, 267, 12397]

Batch (block_size=4):
x: [334, 3855, 288, 267]
y: [3855, 288, 267, 2959]

Training objective:
  Given [334], predict 3855
  Given [334, 3855], predict 288
  Given [334, 3855, 288], predict 267
  Given [334, 3855, 288, 267], predict 2959
```

**Why Memory Mapping?**
- **325M tokens** = ~650MB file
- Without mmap: Load entire file into RAM
- With mmap: OS fetches pages on-demand
- **Result**: Can train on 1TB datasets with 16GB RAM

### 2. Optimizer (`optimizer.py`)

AdamW with intelligent parameter grouping.

```python
def configure_optimizers(model, weight_decay, learning_rate, betas):
    # Separate parameters by dimensionality
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
```

**Parameter Grouping Logic:**

```
2D+ Tensors (Weights):
  ✅ Apply weight decay
  - Embedding weights
  - Linear layer weights
  - Attention projection weights

1D Tensors (Norms/Biases):
  ❌ No weight decay
  - RMSNorm weights
  - Biases (if any)
```

**Why Separate?**
- Weight decay = L2 regularization
- Norms are small and sensitive
- Decaying them hurts performance

### 3. Learning Rate Scheduler (`lr_scheduler.py`)

Cosine decay with linear warmup.

```python
def get_lr(it, args):
    # 1. Warmup phase
    if it < warmup_steps:
        return learning_rate * (it + 1) / (warmup_steps + 1)
    
    # 2. Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)
```

**Learning Rate Curve:**

```
LR
 │     ╱‾‾‾‾‾‾‾‾╲___
 │    ╱            ╲___
 │   ╱                 ╲___
 │__╱_______________________╲___
   0   200        2500      5000  (steps)
   warmup         decay
```

**Why This Schedule?**

1. **Warmup (0-200 steps)**:
   - Prevents exploding gradients
   - Model weights are random initially
   - Large LR would cause instability

2. **Cosine Decay (200-5000 steps)**:
   - Smooth reduction to min_lr
   - Helps model converge
   - Better than step decay

### 4. Training Loop (`train.py`)

Main training script with all components integrated.

**Key Features:**

```python
# 1. Gradient Accumulation
for micro_step in range(gradient_accumulation_steps):
    X, Y = train_loader.get_batch(device)
    logits, _ = model(X)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    loss = loss / gradient_accumulation_steps
    loss.backward()

# 2. Gradient Clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

# 3. Optimizer Step
optimizer.step()
optimizer.zero_grad()
```

**Training Flow Diagram:**

```
┌─────────────────────────────────────────────┐
│ 1. Load Batch (16 × 512 tokens)             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 2. Forward Pass                             │
│    - Embedding                              │
│    - 16 Transformer Blocks                  │
│    - Output Projection                      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 3. Calculate Loss                           │
│    CrossEntropy(logits, targets)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 4. Backward Pass                            │
│    Compute gradients                        │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 5. Gradient Accumulation (×4)               │
│    Effective batch = 16 × 4 = 64            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 6. Clip Gradients (max_norm=1.0)            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 7. Optimizer Step (AdamW)                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 8. Update Learning Rate                     │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 9. Log & Checkpoint (periodic)              │
└─────────────────────────────────────────────┘
```

## Configuration

Training is controlled via YAML config files.

### Production Config (`config.yaml`)

```yaml
# Training
batch_size: 16
block_size: 512
gradient_accumulation_steps: 4
max_iters: 5000

# Learning Rate
learning_rate: 6.0e-4
min_lr: 6.0e-5
warmup_iters: 200

# Optimizer
weight_decay: 0.1
beta1: 0.9
beta2: 0.95
grad_clip: 1.0

# System
device: 'cuda'
compile: true
dtype: 'bfloat16'
```

### Test Config (`config_test.yaml`)

```yaml
# Quick validation
batch_size: 16
block_size: 128
max_iters: 10
device: 'cpu'
compile: false
```

## Usage

### Basic Training

```bash
# Use default config
python run_train.py

# Use custom config
python run_train.py --config train/config.yaml
```

### Training Example

**Output:**
```
Loading data from data/bin...
[train] Data loaded. Length: 325,004,796 tokens
[val] Data loaded. Length: 36,111,644 tokens
Initializing Model...
Optimizer Configured: 113 tensors decayed (50,036,736 params)
Starting training...

iter 0: loss 329.50, time 50.2ms, lr 0.000200
iter 10: loss 282.85, time 48.7ms, lr 0.000400
iter 20: loss 151.14, time 49.1ms, lr 0.000600
...
step 200: train loss 63.22, val loss 69.54
Saving checkpoint to out/ckpt.pt
...
iter 5000: loss 12.34, time 47.8ms, lr 0.000060
Training Complete!
```

## Training Metrics

### Loss Trajectory

```
Loss
 │
300│ ●
   │  ●
200│   ●●
   │     ●●●
100│        ●●●●●
   │            ●●●●●●●
 50│                   ●●●●●●●
   │                         ●●●●●●
 10│_________________________________●●●
    0   1000  2000  3000  4000  5000
                Steps
```

### Perplexity

```
Perplexity = exp(loss)

Initial: exp(329) = ∞ (random)
Step 1000: exp(50) ≈ 5.2e21
Step 5000: exp(12) ≈ 162,754

Lower is better!
```

## Performance Optimization

### 1. Mixed Precision Training

```python
# BF16 on modern GPUs
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

with ctx:
    logits, _ = model(X)
    loss = F.cross_entropy(...)
```

**Speedup**: ~2x faster, ~50% less memory

### 2. Gradient Accumulation

```python
# Simulate batch_size = 64 with only 16 per step
effective_batch = batch_size * gradient_accumulation_steps
# 16 × 4 = 64
```

**Benefit**: Train with large batches on small GPUs

### 3. Torch Compile

```python
if compile:
    model = torch.compile(model)
```

**Speedup**: ~20-30% faster on NVIDIA GPUs

### 4. Fused AdamW

```python
optimizer = torch.optim.AdamW(..., fused=True)
```

**Speedup**: ~10-15% faster optimizer step

## Checkpointing

Checkpoints are saved periodically and contain:

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
}
```

**Resume Training:**

```python
checkpoint = torch.load('out/ckpt.pt')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
start_iter = checkpoint['iter_num']
```

## Hyperparameter Guide

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `batch_size` | 16-32 | Fit to GPU memory |
| `block_size` | 512-2048 | Longer = better context |
| `learning_rate` | 6e-4 | Scale with batch size |
| `warmup_iters` | 200-500 | ~4% of total steps |
| `weight_decay` | 0.1 | Standard for LLMs |
| `grad_clip` | 1.0 | Prevents explosions |

## Training Time Estimates

**On NVIDIA A100 (40GB):**
- **5,000 steps**: ~2-3 hours
- **50,000 steps**: ~20-30 hours
- **500,000 steps**: ~8-12 days

**Tokens Seen:**
```
tokens_per_step = batch_size × block_size × grad_accum_steps
                = 16 × 512 × 4
                = 32,768 tokens/step

5,000 steps = 163M tokens (~0.5% of dataset)
50,000 steps = 1.6B tokens (~5× dataset)
```

## References

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [Gradient Accumulation](https://arxiv.org/abs/1711.00489)
