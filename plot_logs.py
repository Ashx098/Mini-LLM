import json
import matplotlib.pyplot as plt
import os
import argparse

def read_jsonl(filename):
    data = []
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found.")
        return data
    with open(filename, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def plot_logs(log_dir='logs', output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    train_file = os.path.join(log_dir, 'train.jsonl')
    eval_file = os.path.join(log_dir, 'eval.jsonl')
    
    train_data = read_jsonl(train_file)
    eval_data = read_jsonl(eval_file)
    
    if not train_data:
        print("No training data found!")
        return

    # 1. Plot Training Loss
    plt.figure(figsize=(10, 6))
    steps = [d['step'] for d in train_data]
    losses = [d['loss'] for d in train_data]
    plt.plot(steps, losses, label='Train Loss', alpha=0.6)
    
    # Overlay Eval Loss if available
    if eval_data:
        eval_steps = [d['step'] for d in eval_data]
        eval_losses = [d['val_loss'] for d in eval_data]
        plt.plot(eval_steps, eval_losses, 'r-o', label='Val Loss', linewidth=2)
        
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    print(f"Saved {output_dir}/loss_curve.png")
    plt.close()

    # 2. Plot Learning Rate
    plt.figure(figsize=(10, 6))
    lrs = [d['lr'] for d in train_data]
    plt.plot(steps, lrs, color='green', label='Learning Rate')
    plt.xlabel('Steps')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'lr_curve.png'))
    print(f"Saved {output_dir}/lr_curve.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--out_dir', type=str, default='plots')
    args = parser.parse_args()
    
    plot_logs(args.log_dir, args.out_dir)
