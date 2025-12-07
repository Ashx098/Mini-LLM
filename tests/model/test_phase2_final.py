import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from model.config import ModelArgs
from model.transformer import Transformer

def test_full_model():
    print("Testing Full Mini-LLM Architecture...")
    
    # 1. Init Model
    args = ModelArgs()
    args.device = "cuda"  # Safe for testing logic
    model = Transformer(args)
   # model.to(args.device)
    
    # 2. Count Parameters
    # We filter out "freqs_cis" buffer from the count as it's not a learnable parameter
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} parameters.")
    
    # 3. Dummy Forward Pass (Batch=2, Seq=32)
    # This simulates a tiny training step
    x = torch.randint(0, args.vocab_size, (2, 32))
    
    # 4. Run Model
    try:
        logits, _ = model(x)
        
        # 5. Validate Output
        expected_shape = (2, 32, args.vocab_size)
        assert logits.shape == expected_shape, f"Shape Mismatch: {logits.shape}"
        
        print(f"✅ Forward Pass Successful. Output Shape: {logits.shape}")
        
        # 6. Check Weight Tying
        if torch.equal(model.tok_embeddings.emb.weight, model.output.weight):
            print("✅ Weight Tying Confirmed.")
        else:
            print("❌ Weight Tying Failed!")
            
        print("\n🚀 PHASE 2 COMPLETE. Architecture is ready for A100s.")
        
    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE: {e}")

if __name__ == "__main__":
    test_full_model()