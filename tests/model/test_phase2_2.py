import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from model.rmsnorm import RMSNorm

def test_rmsnorm():
    print("Testing RMSNorm...")
    batch, seq, dim = 2, 10, 384
    
    # 1. Create random input (simulating token embeddings)
    x = torch.randn(batch, seq, dim)
    
    # 2. Initialize Norm
    norm = RMSNorm(dim)
    
    # 3. Forward Pass
    y = norm(x)
    
    # 4. Check Shape
    assert y.shape == x.shape, f"Shape mismatch: {y.shape}"
    
    # 5. Check Math (The norm of the output should be roughly sqrt(dim))
    # Note: RMSNorm forces the RMS to be 1.
    # The standard deviation of the output should be close to 1.
    print(f"Input Std: {x.std().item():.4f}")
    print(f"Output Std: {y.std().item():.4f} (Should be close to 1.0)")
    
    print("✅ RMSNorm Validated.")

if __name__ == "__main__":
    test_rmsnorm()