import os
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

def process_data():
    # 1. Config
    input_file_path = "data/raw/merged_text/corpus.txt"  # PATH TO YOUR DATA
    tokenizer_path = "Tokenizer/BPE"                     # PATH TO YOUR NEW TOKENIZER
    output_dir = "data/bin"
    val_split_ratio = 0.1  # 10% for validation

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure eos_token is present (usually ID 2)
    eos_id = tokenizer.eos_token_id
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS ID: {eos_id}")

    # 3. Read Data
    print(f"Reading {input_file_path}...")
    with open(input_file_path, 'r', encoding='utf-8') as f:
        # Read all lines
        lines = f.readlines()
        
    print(f"Total lines: {len(lines):,}")

    # 4. Tokenize
    # We use a simple list comprehension for the 80M scale. 
    # For 100B scale, we would use parallel processing (multiprocessing).
    print("Tokenizing...")
    all_tokens = []
    
    # Using tqdm for progress bar
    for line in tqdm(lines):
        text = line.strip()
        if not text:
            continue
            
        # Encode text and append EOS token
        # This tells the model where one sentence ends and the next begins
        tokens = tokenizer.encode(text)
        tokens.append(eos_id) 
        all_tokens.extend(tokens)

    token_count = len(all_tokens)
    print(f"Total tokens: {token_count:,}")

    # 5. Convert to Numpy (uint16 saves 50% RAM)
    # 32,000 fits easily in uint16 (max 65,535)
    ids = np.array(all_tokens, dtype=np.uint16)

    # 6. Split Train/Val
    val_count = int(token_count * val_split_ratio)
    train_ids = ids[:-val_count]
    val_ids = ids[-val_count:]

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")

    # 7. Save to disk (Memory Mapped friendly)
    train_ids.tofile(os.path.join(output_dir, "train.bin"))
    val_ids.tofile(os.path.join(output_dir, "val.bin"))
    
    print(f"✅ Saved binary files to {output_dir}/")

if __name__ == "__main__":
    process_data()
