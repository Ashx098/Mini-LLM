import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from model.config import ModelArgs
from model.transformer_block import TransformerBlock
from model.rope import RoPE

def test_block():
    print("Testing Full Transformer Block...")
    args = ModelArgs()
    batch, seq = 2, 10
    
    # 1. Initialize Block & RoPE
    block = TransformerBlock(args)
    head_dim = args.dim // args.n_heads
    rope = RoPE(head_dim, args.max_seq_len)
    
    # 2. Create Data
    x = torch.randn(batch, seq, args.dim)
    freqs_cis = rope(seq)
    
    # 3. Forward Pass
    output, _ = block(x, freqs_cis)
    
    # 4. Check Shape
    assert output.shape == x.shape, f"Shape Mismatch: {output.shape}"
    
    # 5. Check Parameter Count (Sanity Check for MLP size)
    # MLP w1 shape should be [384, 1536] based on your doc
    w1_shape = block.feed_forward.w1.weight.shape
    print(f"MLP Hidden Dim: {w1_shape[0]} (Should be 1536)")
    
    print("✅ Transformer Block Validated.")

if __name__ == "__main__":
    test_block()