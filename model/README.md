# Model Architecture

This module contains the core transformer architecture for Mini-LLM, implementing a decoder-only language model similar to GPT and LLaMA.

## Overview

**Mini-LLM** is an 80M parameter transformer model featuring:
- **RoPE** (Rotary Position Embeddings) for position encoding
- **RMSNorm** for layer normalization
- **SwiGLU** activation in feedforward layers
- **Grouped Query Attention** (GQA) support
- **Weight Tying** between embeddings and output layer

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Token IDs                         │
│                    [Batch, Seq_Len]                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Token Embedding (384 dim)                       │
│              + Dropout                                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   Transformer Block × 16     │
        │  ┌────────────────────────┐  │
        │  │  RMSNorm               │  │
        │  │  ↓                     │  │
        │  │  Multi-Head Attention  │  │
        │  │  (with RoPE)           │  │
        │  │  ↓                     │  │
        │  │  Residual Add          │  │
        │  │  ↓                     │  │
        │  │  RMSNorm               │  │
        │  │  ↓                     │  │
        │  │  SwiGLU FFN            │  │
        │  │  ↓                     │  │
        │  │  Residual Add          │  │
        │  └────────────────────────┘  │
        └──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final RMSNorm                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Output Projection (32K vocab)                   │
│              [Batch, Seq_Len, 32000]                         │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Token Embedding (`embedding.py`)

Converts token IDs to dense vectors.

```python
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        self.emb = nn.Embedding(vocab_size, dim)
    
    def forward(self, x):
        # x: [Batch, Seq] → [Batch, Seq, Dim]
        return self.emb(x)
```

**Example:**
```
Input:  [1, 334, 3855]  (token IDs)
Output: [[0.12, -0.34, ...],   # 384-dim vector for token 1
         [0.56, 0.78, ...],    # 384-dim vector for token 334
         [-0.23, 0.45, ...]]   # 384-dim vector for token 3855
```

### 2. RMSNorm (`rmsnorm.py`)

Root Mean Square Layer Normalization - faster and more stable than LayerNorm.

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        # 1. Calculate RMS
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        # 2. Normalize
        x_normed = x * rms
        # 3. Scale by learnable weight
        return x_normed * self.weight
```

**Why RMSNorm?**
- No mean centering (faster)
- Better gradient flow
- Used in LLaMA, GPT-NeoX

### 3. RoPE (`rope.py`)

Rotary Position Embeddings - encodes position information through rotation.

```python
# Pre-compute rotation frequencies
freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
t = torch.arange(max_seq_len)
freqs_cis = torch.polar(torch.ones_like(freqs), torch.outer(t, freqs))

# Apply rotation to Q and K
xq_complex = torch.view_as_complex(xq.reshape(..., -1, 2))
xq_rotated = xq_complex * freqs_cis
```

**Visual Example:**

```
Position 0:  Q₀ rotated by 0°
Position 1:  Q₁ rotated by θ
Position 2:  Q₂ rotated by 2θ
Position 3:  Q₃ rotated by 3θ

Relative position = difference in rotation angle
→ Model learns: "word at position i+k is k steps away"
```

**Advantages:**
- Relative position encoding
- Extrapolates to longer sequences
- No learned position embeddings needed

### 4. Attention (`attention.py`)

Multi-head self-attention with optional Grouped Query Attention (GQA).

```python
class Attention(nn.Module):
    def forward(self, x, freqs_cis):
        # 1. Project to Q, K, V
        xq = self.wq(x)  # [B, S, n_heads * head_dim]
        xk = self.wk(x)  # [B, S, n_kv_heads * head_dim]
        xv = self.wv(x)
        
        # 2. Apply RoPE to Q and K
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # 3. Scaled Dot-Product Attention
        scores = (xq @ xk.T) / sqrt(head_dim)
        attn = softmax(scores, dim=-1)
        output = attn @ xv
        
        # 4. Output projection
        return self.wo(output)
```

**Attention Flow:**

```
Input: "The cat sat on the mat"

Q: "mat" wants to attend to all words
K: All words offer their "keys"
V: All words offer their "values"

Attention Scores:
mat → The: 0.05
mat → cat: 0.10
mat → sat: 0.15
mat → on:  0.20
mat → the: 0.25
mat → mat: 0.25  ← highest (self-attention)

Output: weighted sum of values
```

### 5. SwiGLU FFN (`mlp.py`)

Feedforward network with SwiGLU activation (used in LLaMA, PaLM).

```python
class FeedForward(nn.Module):
    def forward(self, x):
        # SwiGLU: silu(W1(x)) ⊙ W3(x)
        gate = F.silu(self.w1(x))  # Smooth activation
        value = self.w3(x)
        return self.w2(gate * value)  # Element-wise multiply
```

**Dimensions:**
```
Input:  [B, S, 384]
  ↓ W1, W3
Hidden: [B, S, 1536]  (4× expansion)
  ↓ W2
Output: [B, S, 384]
```

**Why SwiGLU?**
- Better than ReLU/GELU
- Gating mechanism (like LSTM)
- Empirically improves performance

### 6. Transformer Block (`transformer_block.py`)

Combines attention and FFN with residual connections.

```python
class TransformerBlock(nn.Module):
    def forward(self, x, freqs_cis):
        # Attention sub-layer
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        
        # FFN sub-layer
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out
```

**Pre-Norm Architecture:**
```
x → Norm → Attention → + → Norm → FFN → + → output
    ↑__________________|     ↑___________|
       Residual Connection      Residual Connection
```

### 7. Full Transformer (`transformer.py`)

Assembles all components into the complete model.

```python
class Transformer(nn.Module):
    def __init__(self, args):
        self.tok_embeddings = TokenEmbedding(...)
        self.layers = nn.ModuleList([
            TransformerBlock(args) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        
        # Weight tying
        self.output.weight = self.tok_embeddings.emb.weight
```

## Model Configuration

**File**: `config.py`

```python
@dataclass
class ModelArgs:
    dim: int = 384           # Hidden dimension
    n_layers: int = 16       # Number of blocks
    n_heads: int = 6         # Attention heads
    n_kv_heads: int = 6      # KV heads (for GQA)
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.0
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Token Embeddings | 12.3M |
| Transformer Blocks (×16) | 37.7M |
| Final Norm | 384 |
| Output (tied) | 0 |
| **Total** | **~50M** |

## Example: Forward Pass

**Input:**
```python
tokens = torch.tensor([[1, 334, 3855, 288]])  # "The cat sat"
```

**Processing:**

```
Step 1: Embedding
  [1, 334, 3855, 288] → [1, 4, 384]

Step 2: Layer 0
  RMSNorm → Attention (with RoPE) → Add
  RMSNorm → FFN (SwiGLU) → Add

Step 3: Layers 1-15
  (repeat)

Step 4: Final Norm
  [1, 4, 384] → [1, 4, 384]

Step 5: Output Projection
  [1, 4, 384] → [1, 4, 32000]  (logits for each token)
```

**Output:**
```
Logits for next token prediction:
Position 0 ("The"): [0.12, -0.34, ..., 0.56]  (32K values)
Position 1 ("cat"): [-0.23, 0.45, ..., -0.12]
Position 2 ("sat"): [0.67, -0.89, ..., 0.34]
Position 3 ("on"):  [0.23, 0.12, ..., -0.45]
```

## Key Design Choices

1. **RoPE over Learned PE**: Better extrapolation to longer sequences
2. **RMSNorm over LayerNorm**: Faster, simpler, equally effective
3. **SwiGLU over GELU**: Empirically better performance
4. **Pre-Norm**: More stable training than Post-Norm
5. **Weight Tying**: Reduces parameters, improves low-resource performance

## References

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
