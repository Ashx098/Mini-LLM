import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rope import apply_rotary_emb

class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        # GQA: How many times to repeat the KV heads to match Q heads?
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # Projections
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.dropout = args.dropout

    def forward(self, x, freqs_cis, mask=None, use_cache=False, past_kv=None):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project Q, K, V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # 2. Reshape for heads: (Batch, Seq, Heads, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # 3. Apply RoPE (Rotates Q and K vectors)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # 4. Handle KV Cache (For Inference Step 5 later)
        if use_cache:
            if past_kv is not None:
                xk_past, xv_past = past_kv
                xk = torch.cat([xk_past, xk], dim=1)
                xv = torch.cat([xv_past, xv], dim=1)
            # Save current state for next token
            current_kv = (xk, xv)
        else:
            current_kv = None
            
        # 5. GQA Expansion: If n_kv_heads < n_heads, repeat K/V data
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)
            
        # 6. Transpose for Attention: (Batch, Heads, Seq, Head_Dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # 7. Flash Attention (The engine!)
        # is_causal=True ensures the model can't look at future tokens (Masking)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, 
            attn_mask=mask, 
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True if mask is None else False 
        )
        
        # 8. Reshape back: (Batch, Seq, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # 9. Final Projection
        return self.wo(output), current_kv
        