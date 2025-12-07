import os
import sys
import time
import math
import torch
import numpy as np
import yaml
import argparse
import datetime
from model.config import ModelArgs
from model.transformer import Transformer
from train.dataloader import DataLoader
from train.optimizer import configure_optimizers
from train.lr_scheduler import get_lr

# -----------------------------------------------------------------------------
# Load Configuration from YAML
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Train Mini-LLM')
parser.add_argument('--config', type=str, default='train/config_test.yaml',
                    help='Path to config file (default: train/config_test.yaml)')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

print(f"Loading configuration from: {args.config}")
print(f"Config: {config}")

# Extract config values
out_dir = config['out_dir']
eval_interval = config['eval_interval']
log_interval = config['log_interval']
eval_iters = config['eval_iters']
always_save_checkpoint = config['always_save_checkpoint']

wandb_log = config['wandb_log']
wandb_project = config['wandb_project']
wandb_run_name = config['wandb_run_name']

batch_size = config['batch_size']
block_size = config['block_size']
gradient_accumulation_steps = config['gradient_accumulation_steps']
max_iters = config['max_iters']
lr_decay_iters = config['lr_decay_iters']
min_lr = config['min_lr']
learning_rate = config['learning_rate']
warmup_iters = config['warmup_iters']

weight_decay = config['weight_decay']
beta1 = config['beta1']
beta2 = config['beta2']
grad_clip = config['grad_clip']

device = config['device']
compile_model = config['compile']
dtype = config['dtype']
data_dir = config['data_dir']
seed = config['seed']

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

torch.manual_seed(seed)
os.makedirs(out_dir, exist_ok=True)
ctx = torch.amp.autocast(device_type=device, dtype=getattr(torch, dtype))

# 1. Initialize DataLoaders
print(f"Loading data from {data_dir}...")
train_loader = DataLoader(data_dir, batch_size, block_size, split="train")
val_loader = DataLoader(data_dir, batch_size, block_size, split="val")

# 2. Initialize Model
print("Initializing Model...")
model_args = ModelArgs(
    dim=config['model']['dim'],
    n_layers=config['model']['n_layers'],
    n_heads=config['model']['n_heads'],
    n_kv_heads=config['model']['n_kv_heads'],
    vocab_size=config['model']['vocab_size'],
    multiple_of=config['model']['multiple_of'],
    max_seq_len=config['model']['max_seq_len'],
    dropout=config['model']['dropout']
)
model = Transformer(model_args)
model.to(device)

# 3. Optimizer
optimizer = configure_optimizers(model, weight_decay=weight_decay, learning_rate=learning_rate, betas=(beta1, beta2), device_type=device)

# 4. Compile (Speedup for NVIDIA GPUs)
if compile_model:
    print("Compiling model... (This takes a minute but makes training faster)")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.get_batch(device)
            with ctx:
                logits, _ = model(X)
                # Calculate Loss (CrossEntropy)
                # Flatten: (Batch * Seq, Vocab)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
import json
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
train_log_file = os.path.join(log_dir, 'train.jsonl')
eval_log_file = os.path.join(log_dir, 'eval.jsonl')

print(f"Logging to {log_dir}...")

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
print("Starting training...")
t0 = time.time()
iter_num = 0
best_val_loss = 1e9

while iter_num < max_iters:
    # 1. Determine Learning Rate
    lr = get_lr(iter_num, args=type('Args', (object,), {
        'learning_rate': learning_rate, 'min_lr': min_lr,
        'warmup_steps': warmup_iters, 'max_steps': lr_decay_iters
    }))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 2. Evaluation & Checkpointing
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Log Eval Stats
        with open(eval_log_file, 'a') as f:
            f.write(json.dumps({
                'step': iter_num,
                'train_loss': losses['train'].item(),
                'val_loss': losses['val'].item(),
                'timestamp': time.time()
            }) + '\n')
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
            }
            print(f"Saving checkpoint to {out_dir}/ckpt.pt")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    # 3. Forward Backward Update (Gradient Accumulation)
    for micro_step in range(gradient_accumulation_steps):
        X, Y = train_loader.get_batch(device)
        with ctx:
            logits, _ = model(X)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            loss = loss / gradient_accumulation_steps # Scale loss
        
        loss.backward()

    # 4. Optimizer Step
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip) # Prevent explosions
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # 5. Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        
        # Calculate ETA
        remaining_steps = max_iters - iter_num
        eta_seconds = remaining_steps * dt
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
        
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.6f}, eta {eta_str}")
        
        # Log Train Stats
        with open(train_log_file, 'a') as f:
            f.write(json.dumps({
                'step': iter_num,
                'loss': lossf,
                'lr': lr,
                'time_ms': dt * 1000,
                'eta_seconds': eta_seconds,
                'eta': eta_str,
                'timestamp': time.time()
            }) + '\n')
            
        if wandb_log:
            import wandb
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "lr": lr,
                "time_ms": dt * 1000,
                "eta_hours": eta_seconds / 3600
            })
        
    iter_num += 1

print("Training Complete!")
