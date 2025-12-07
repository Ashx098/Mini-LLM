import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from model.rope import RoPE, apply_rotary_emb

def test_rope():
    print("Testing RoPE...")
    dim = 64
    seq_len = 10
    batch = 2
    n_heads = 4
    
    # 1. Init RoPE
    rope = RoPE(dim=dim, max_seq_len=seq_len)
    freqs_cis = rope(seq_len)
    
    print(f"Frequencies Shape: {freqs_cis.shape} (Should be [10, 32])")
    
    # 2. Create Dummy Queries (Batch, Seq, Heads, Dim)
    xq = torch.randn(batch, seq_len, n_heads, dim)
    xk = torch.randn(batch, seq_len, n_heads, dim)
    
    # 3. Save original magnitude (norm)
    orig_norm = xq.norm()
    
    # 4. Apply Rotation
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)
    
    # 5. Check Output Shape
    assert xq_rot.shape == xq.shape, f"Shape mismatch: {xq_rot.shape}"
    
    # 6. Check Magnitude Preservation
    # Rotation simply spins the vector, it shouldn't shrink or grow it.
    new_norm = xq_rot.norm()
    diff = abs(orig_norm - new_norm).item()
    
    print(f"Original Norm: {orig_norm:.4f}")
    print(f"Rotated Norm:  {new_norm:.4f}")
    print(f"Difference:    {diff:.6f}")
    
    if diff < 1e-3:
        print("✅ RoPE Validated (Magnitude preserved).")
    else:
        print("❌ RoPE Failed (Magnitude changed!)")

if __name__ == "__main__":
    test_rope()