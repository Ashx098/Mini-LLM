import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The learnable weight parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 1. Calculate the Root Mean Square (RMS)
        # x.pow(2) -> squares every number
        # .mean(-1, keepdim=True) -> averages them
        # .rsqrt() -> takes 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 2. Critical Stability Hack: Always cast to float32 for the division
        # This prevents "NaN" (Not a Number) errors in half-precision training
        output = self._norm(x.float()).type_as(x)
        
        # 3. Scale by the learnable weight
        return output * self.weight