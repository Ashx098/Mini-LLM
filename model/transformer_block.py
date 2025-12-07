import torch
import torch.nn as nn
from model.attention import Attention
from model.mlp import FeedForward
from model.rmsnorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        
        # 1. Attention Unit
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim)
        
        # 2. MLP Unit
        self.feed_forward = FeedForward(args)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x, freqs_cis, mask=None, use_cache=False, past_kv=None):
        # 1. Attention Block
        # Residual Connection: x + Attention(Norm(x))
        h_attn, current_kv = self.attention(
            self.attention_norm(x), 
            freqs_cis=freqs_cis, 
            mask=mask, 
            use_cache=use_cache, 
            past_kv=past_kv
        )
        h = x + h_attn
        
        # 2. FeedForward Block
        # Residual Connection: h + MLP(Norm(h))
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out, current_kv