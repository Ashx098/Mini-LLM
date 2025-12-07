from dataclasses import dataclass
import torch

@dataclass
class ModelArgs:
    dim: int = 384          # Hidden dimension size (d_model)
    n_layers: int = 16      # Number of Transformer blocks
    n_heads: int = 6        # Number of Attention Heads
    n_kv_heads: int = 6     # Number of KV Heads (for GQA later)
    vocab_size: int = 32000 # Matches your SentencePiece model
    multiple_of: int = 32   # Helps SwiGLU hidden layer size stay efficient
    max_seq_len: int = 2048 # Context Window
    dropout: float = 0.0    # No dropout for pre-training usually
    device: str = "cuda" if torch.cuda.is_available() else "cpu"