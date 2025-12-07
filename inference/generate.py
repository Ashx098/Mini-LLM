import torch
import torch.nn.functional as F
from model.config import ModelArgs
from model.transformer import Transformer
from transformers import AutoTokenizer
from inference.sampling import apply_temperature, sample_top_p
import os

class Generator:
    def __init__(self, checkpoint_path, tokenizer_path, device="cuda"):
        self.device = device
        
        # 1. Load Tokenizer
        print(f"Loading Tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 2. Load Model Config & Weights
        print(f"Loading Checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Reconstruct Args from checkpoint (ensures config matches training)
        self.args = checkpoint['model_args']
        self.args.device = device
        
        # Initialize Model
        self.model = Transformer(self.args)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.model.eval() # Important! Turns off Dropout

        print("Generator Ready.")

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
        # 1. Tokenize Prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Start sequence
        curr_ids = input_ids
        
        # KV Cache container
        past_kv = None 
        
        # 2. Generation Loop
        for _ in range(max_new_tokens):
            # We only pass the *last* token if we have a cache
            # On first step, we pass the whole prompt
            if past_kv is None:
                x = curr_ids
                start_pos = 0
            else:
                x = curr_ids[:, -1:] # Just the last generated token
                start_pos = curr_ids.shape[1] - 1 # Position index
            
            # 3. Forward Pass
            logits, past_kv = self.model(x, start_pos=start_pos, use_cache=True, past_kv=past_kv)
            
            # 4. Sampling
            # Get logits for the last token
            next_token_logits = logits[:, -1, :]
            
            # Apply Temp
            next_token_logits = apply_temperature(next_token_logits, temperature)
            
            # Apply Softmax to get probs
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Apply Top-P
            next_token = sample_top_p(probs, top_p)
            
            # 5. Append & Print
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Optional: Stop if EOS token is generated
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        # 6. Decode
        output_text = self.tokenizer.decode(curr_ids[0].tolist(), skip_special_tokens=True)
        return output_text

# Simple CLI test
if __name__ == "__main__":
    # Point to your checkpoint (We will use the dry-run checkpoint or a fresh one)
    # Since dry-run didn't save, we mock a save or you need to re-run train with max_iters=5 and eval_interval=5
    
    CHECKPOINT_PATH = "out/ckpt.pt" 
    TOKENIZER_PATH = "Tokenizer/BPE"
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"⚠️ Checkpoint {CHECKPOINT_PATH} not found!")
        print("Please run 'python train/train.py' again with 'max_iters=5' and 'eval_interval=5' to generate it.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = Generator(CHECKPOINT_PATH, TOKENIZER_PATH, device=device)
        
        prompt = "Hello world"
        print(f"\nPrompt: {prompt}")
        print("-" * 30)
        output = gen.generate(prompt, max_new_tokens=20)
        print(output)
        print("-" * 30)
