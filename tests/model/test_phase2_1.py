import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from model.config import ModelArgs
from model.embedding import TokenEmbedding

def test_embedding():
    # 1. Initialize Config
    config = ModelArgs()
    print(f"Loaded Config: Dim={config.dim}, Vocab={config.vocab_size}")

    # 2. Initialize Embedding Layer
    emb_layer = TokenEmbedding(config.vocab_size, config.dim)
    print("Embedding Layer Initialized.")

    # 3. Create Dummy Data (Batch Size=2, Sequence Length=4)
    dummy_input = torch.tensor([[1, 500, 20, 2], [1, 999, 5, 2]], dtype=torch.long)
    
    # 4. Forward Pass
    output = emb_layer(dummy_input)

    # 5. Check Shapes
    expected_shape = (2, 4, 384) # (Batch, Seq, Dim)
    assert output.shape == expected_shape, f"Shape mismatch! Got {output.shape}"
    
    print(f"✅ Success! Output Shape: {output.shape}")
    print("Step 2.1 & 2.9 Validated.")

if __name__ == "__main__":
    test_embedding()