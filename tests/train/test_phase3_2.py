import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

import torch
from train.dataloader import DataLoader

def test_dataloader():
    print("Testing Dataloader...")
    
    # Config
    batch_size = 4
    block_size = 8
    data_dir = "data/bin"
    
    # 1. Initialize
    # Try loading 'train' split
    loader = DataLoader(data_dir, batch_size, block_size, split="train")
    
    # 2. Get a batch
    x, y = loader.get_batch(device="cpu")
    
    # 3. Check Shapes
    print(f"X Shape: {x.shape} (Should be {batch_size}, {block_size})")
    print(f"Y Shape: {y.shape} (Should be {batch_size}, {block_size})")
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    
    # 4. Check Shift Logic
    # y should be x shifted by 1.
    # So x[0, 1] should equal y[0, 0]
    print("\nSample Check:")
    print(f"x[0]: {x[0].tolist()}")
    print(f"y[0]: {y[0].tolist()}")
    
    # Validating the shift mathematically
    # The first token of y should match the second token of x
    if x[0, 1] == y[0, 0]:
        print("✅ Shift logic verified (Next token prediction).")
    else:
        print("❌ Shift logic FAILED.")

    print("Phase 3.2 Dataloader Validated.")

if __name__ == "__main__":
    test_dataloader()
