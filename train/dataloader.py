import torch
import numpy as np
import os

class DataLoader:
    def __init__(self, data_dir, batch_size, block_size, split="train"):
        self.batch_size = batch_size
        self.block_size = block_size
        
        # 1. Construct filename
        filename = os.path.join(data_dir, f"{split}.bin")
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
            
        # 2. Memory Map the binary file
        # 'r' mode = Read-only (safe for training)
        # dtype=np.uint16 matches the format we saved in tokenize.py
        self.data = np.memmap(filename, dtype=np.uint16, mode='r')
        print(f"[{split}] Data loaded. Length: {len(self.data):,} tokens")

    def get_batch(self, device="cpu"):
        # 3. Random Sampling
        # We need a chunk of length (block_size + 1) to get both x and y
        # ix will be a list of random starting positions
        high = len(self.data) - self.block_size
        ix = torch.randint(low=0, high=high, size=(self.batch_size,))
        
        # 4. Stack batch
        # We convert to int64 (Long) because PyTorch embedding layers expect LongTensors
        x = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1 : i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        
        # 5. Move to Device (GPU/CPU)
        if device == "cuda":
            # pin_memory=True logic is usually handled in the loop, 
            # but for simple implementations we move immediately.
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            
        return x, y
