import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from model.config import ModelArgs
from model.attention import Attention
from model.rope import RoPE

def test_attention():
    print("Testing Attention Mechanism...")
    
    # 1. Setup Config
    args = ModelArgs()
    batch, seq_len = 2, 10
    
    # 2. Init Modules
    attn = Attention(args)
    rope = RoPE(args.dim // args.n_heads, args.max_seq_len)
    
    # 3. Create Inputs
    x = torch.randn(batch, seq_len, args.dim)
    
    # 4. Get RoPE Frequencies for this sequence length
    freqs_cis = rope(seq_len)
    
    # 5. Forward Pass (No Cache)
    output, cache = attn(x, freqs_cis)
    
    # 6. Validation
    # Output shape must match input shape (B, S, D)
    assert output.shape == (batch, seq_len, args.dim), f"Shape Mismatch: {output.shape}"
    
    # Cache should be None since we didn't enable it
    assert cache is None
    
    print("✅ Standard Forward Pass: Success")
    
    # 7. Test KV Cache (Simulating generation step)
    print("Testing KV Cache...")
    # Simulate processing just 1 new token
    x_new = torch.randn(batch, 1, args.dim)
    freqs_cis_new = rope(seq_len + 1)[-1:] # Get freq for position 11
    
    # Create fake past cache (Batch, Seq=10, Heads, HeadDim)
    head_dim = args.dim // args.n_heads
    past_k = torch.randn(batch, seq_len, args.n_kv_heads, head_dim)
    past_v = torch.randn(batch, seq_len, args.n_kv_heads, head_dim)
    
    output_new, (new_k, new_v) = attn(x_new, freqs_cis_new, use_cache=True, past_kv=(past_k, past_v))
    
    # New Cache should be length 11 (10 past + 1 new)
    assert new_k.shape[1] == 11, f"Cache update failed: {new_k.shape[1]}"
    
    print("✅ KV Cache Update: Success")
    print("Step 2.4 Attention Validated.")

if __name__ == "__main__":
    test_attention()