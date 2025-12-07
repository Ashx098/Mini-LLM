import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Standard hidden size is usually 4 * dim
        # Llama sometimes uses 2/3rds ratio, but your doc math specifies 1536 (4*384)
        hidden_dim = 4 * args.dim
        hidden_dim = int(hidden_dim)
        
        # Ensure hidden_dim is a multiple of 'multiple_of' (usually 32 or 256) for efficiency
        if args.multiple_of > 1:
            hidden_dim = ((hidden_dim + args.multiple_of - 1) // args.multiple_of) * args.multiple_of
            
        # W1: Gate Projection
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        # W2: Output Projection (Down)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        # W3: Value Projection (Up)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU Logic: 
        # 1. Calculate Gate (w1) and Value (w3)
        # 2. Apply SiLU (Swish) activation to Gate
        # 3. Element-wise multiply Gate * Value
        # 4. Project back down (w2)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))