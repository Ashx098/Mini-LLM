import torch
import os
import json
import argparse
from safetensors.torch import save_file
from model.config import ModelArgs

def convert_to_safetensors(checkpoint_path, out_dir):
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return

    # Allow ModelArgs to be loaded safely
    torch.serialization.add_safe_globals([ModelArgs])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    model_state_dict = checkpoint['model']
    model_args = checkpoint['model_args']
    
    # Fix for shared weights (safetensors doesn't like shared memory)
    # We clone the output weight so it's physically separate in memory
    if 'output.weight' in model_state_dict and 'tok_embeddings.emb.weight' in model_state_dict:
        # Check if they actually share memory
        if model_state_dict['output.weight'].data_ptr() == model_state_dict['tok_embeddings.emb.weight'].data_ptr():
            print("   - Detected shared weights. Cloning 'output.weight' for safetensors compatibility...")
            model_state_dict['output.weight'] = model_state_dict['output.weight'].clone()
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Save Weights (safetensors)
    safetensors_path = os.path.join(out_dir, 'model.safetensors')
    print(f"Saving weights to {safetensors_path}...")
    save_file(model_state_dict, safetensors_path)
    
    # 2. Save Config (json)
    config_path = os.path.join(out_dir, 'config.json')
    print(f"Saving config to {config_path}...")
    
    # Convert dataclass to dict
    config_dict = {k: v for k, v in model_args.__dict__.items()}
    # Add architecture metadata
    config_dict['architectures'] = ["MiniLLM"]
    config_dict['model_type'] = "mini-llm"
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
        
    print("✅ Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='out_production/ckpt.pt')
    parser.add_argument('--out_dir', type=str, default='out_safetensors')
    args = parser.parse_args()
    
    convert_to_safetensors(args.checkpoint, args.out_dir)
