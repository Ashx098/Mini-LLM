import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        
        # 1. Calculate the rotation frequencies
        # Formula: theta ^ (-2i / dim)
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        
        # 2. Create position grid (0, 1, 2, ..., max_seq_len)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        # 3. Outer product -> matrix of angles
        freqs = torch.outer(t, freqs)  # shape: (max_seq_len, dim/2)
        
        # 4. Convert to polar form (Complex numbers: cos + i*sin)
        # We use ones as magnitude because we only want rotation, not scaling
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        # Register as a buffer so it's saved with the model but not trained
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, seq_len):
        # Return the pre-computed rotation values for the current sequence length
        return self.freqs_cis[:seq_len]

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq: (batch, seq, n_heads, head_dim)
    
    # 1. Reshape real data into complex pairs (head_dim -> head_dim/2)
    # Example: [x1, x2, x3, x4] -> [(x1,x2), (x3,x4)]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. Reshape frequencies to match batch/head dimensions for broadcasting
    # freqs_cis: (seq, head_dim/2) -> (1, seq, 1, head_dim/2)
    ndim = xq_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    
    # 3. Apply Rotation (Complex Multiplication)
    # (a+bi) * (c+di) performs the rotation logic automatically
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)