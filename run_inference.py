import torch
import os
import argparse
from inference.generate import Generator

def main():
    parser = argparse.ArgumentParser(description='Run inference with Mini-LLM')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Input prompt')
    parser.add_argument('--checkpoint', type=str, default='out_production/ckpt.pt', help='Path to checkpoint')
    parser.add_argument('--tokenizer', type=str, default='Tokenizer/BPE', help='Path to tokenizer')
    parser.add_argument('--temp', type=float, default=0.8, help='Temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='Top-p sampling')
    parser.add_argument('--max_tokens', type=int, default=100, help='Max new tokens')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading model from {args.checkpoint}...")
    gen = Generator(args.checkpoint, args.tokenizer, args.device)
    
    print(f"\nGenerating with prompt: '{args.prompt}'")
    print("-" * 50)
    
    output = gen.generate(
        args.prompt, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temp, 
        top_p=args.top_p
    )
    
    print(output)
    print("-" * 50)

if __name__ == "__main__":
    main()
