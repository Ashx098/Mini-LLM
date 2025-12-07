import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.dim = dim
        
    def forward(self, x):
        # x: [Batch, Seq_Len] (Integer IDs)
        # Returns: [Batch, Seq_Len, Dim] (Float Vectors)
        return self.emb(x)