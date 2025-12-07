import json
import sys
import os

# Default to logs/train.jsonl
log_file = sys.argv[1] if len(sys.argv) > 1 else "logs/train.jsonl"

steps = []
losses = []

if not os.path.exists(log_file):
    print(f"Error: {log_file} not found.")
    print("Usage: python check_loss.py [logs/train.jsonl OR logs/eval.jsonl]")
    exit()

print(f"Reading {log_file}...")
with open(log_file, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'step' in data:
                steps.append(data['step'])
                # Handle both 'loss' (train) and 'val_loss' (eval)
                if 'loss' in data:
                    losses.append(data['loss'])
                elif 'val_loss' in data:
                    losses.append(data['val_loss'])
        except json.JSONDecodeError:
            continue

if not steps:
    print("No valid JSON logs found!")
    exit()

# Simple ASCII Plotter
print(f"\nTraining Progress ({len(steps)} data points):")
print("-" * 60)

# Dynamic scaling
min_loss = min(losses)
max_loss = max(losses)
if max_loss == min_loss:
    max_loss += 1e-6

width = 50
stride = max(1, len(steps) // 20)

for i in range(0, len(steps), stride): 
    step = steps[i]
    loss = losses[i]
    
    # Normalize loss to 0-1 range
    normalized = (loss - min_loss) / (max_loss - min_loss)
    pos = int(normalized * width)
    bar = "#" * pos
    
    print(f"Step {step:5d} | {loss:8.4f} | {bar}")

print("-" * 60)
print(f"Min Loss: {min_loss:.4f}")
print(f"Max Loss: {max_loss:.4f}")
print(f"Current Loss: {losses[-1]:.4f}")
