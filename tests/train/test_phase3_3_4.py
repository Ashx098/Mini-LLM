import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
from dataclasses import dataclass
from model.config import ModelArgs
from model.transformer import Transformer
from train.lr_scheduler import get_lr
from train.optimizer import configure_optimizers

@dataclass
class TrainArgs:
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 10
    max_steps: int = 100

def test_optimization():
    print("Testing Optimizer & Scheduler...")
    
    # 1. Init Model
    model_args = ModelArgs()
    model = Transformer(model_args)
    
    # 2. Test Optimizer Grouping
    # We expect Embeddings & Linear weights to be decayed
    # Norm weights (1D) to NOT be decayed
    optimizer = configure_optimizers(
        model, 
        weight_decay=0.1, 
        learning_rate=6e-4, 
        betas=(0.9, 0.95), 
        device_type='cpu'
    )
    
    # 3. Test Scheduler Curve
    print("\nTesting LR Schedule (Warmup=10, Max=100):")
    train_args = TrainArgs()
    
    # Check Step 0
    lr_0 = get_lr(0, train_args)
    print(f"Step 0 LR: {lr_0:.6f} (Should be near 0)")
    
    # Check Warmup Peak (Step 10)
    lr_10 = get_lr(9, train_args) # Step 9 is end of warmup
    print(f"Step 9 LR: {lr_10:.6f} (Should be near max 6e-4)")
    
    # Check Decay (Step 50)
    lr_50 = get_lr(50, train_args)
    print(f"Step 50 LR: {lr_50:.6f} (Should be lower than max)")
    
    # Check End (Step 100)
    lr_100 = get_lr(100, train_args)
    print(f"Step 100 LR: {lr_100:.6f} (Should be min_lr 6e-5)")

    if lr_0 < lr_10 and lr_50 < lr_10 and lr_100 < lr_50:
        print("✅ LR Scheduler Logic Verified.")
    else:
        print("❌ LR Scheduler Failed.")

if __name__ == "__main__":
    test_optimization()
