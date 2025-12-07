# Data Module

This module handles all data preprocessing, tokenization, and preparation for training.

## Overview

The data pipeline converts raw text into binary token files optimized for training:
- **Raw text collection** from multiple sources
- **Tokenization** using BPE tokenizer
- **Binary serialization** for efficient loading
- **Train/validation splitting**

## Directory Structure

```
data/
├── raw/                    # Raw text sources
│   ├── books/             # Book corpus
│   ├── wikipedia/         # Wikipedia dumps
│   ├── fineweb/           # Web crawl data
│   └── merged_text/
│       └── corpus.txt     # Combined corpus
├── bin/                   # Tokenized binary files
│   ├── train.bin         # Training data (uint16)
│   └── val.bin           # Validation data (uint16)
└── prepare_data.py       # Tokenization script
```

## Data Processing Pipeline

```
┌─────────────────────────────────────────────┐
│ 1. Raw Text Sources                         │
│    - Books: 15 files                        │
│    - Wikipedia: 3 dumps                     │
│    - FineWeb: 1 crawl                       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 2. Merge & Clean                            │
│    → corpus.txt (all text combined)         │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 3. Tokenize (prepare_data.py)              │
│    - Load BPE tokenizer                     │
│    - Process line-by-line                   │
│    - Append EOS tokens                      │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 4. Convert to NumPy (uint16)               │
│    - Vocab size: 32,000 fits in uint16     │
│    - Memory efficient (2 bytes/token)       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│ 5. Train/Val Split (90/10)                 │
│    - train.bin: 325M tokens                 │
│    - val.bin: 36M tokens                    │
└─────────────────────────────────────────────┘
```

## Data Preparation Script

**File**: `prepare_data.py`

```python
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Tokenizer/BPE")
eos_id = tokenizer.eos_token_id

# 2. Read corpus
with open("data/raw/merged_text/corpus.txt") as f:
    lines = f.readlines()

# 3. Tokenize
all_tokens = []
for line in tqdm(lines):
    tokens = tokenizer.encode(line.strip())
    tokens.append(eos_id)  # Mark end of line
    all_tokens.extend(tokens)

# 4. Convert to uint16
ids = np.array(all_tokens, dtype=np.uint16)

# 5. Split
val_count = int(len(ids) * 0.1)
train_ids = ids[:-val_count]
val_ids = ids[-val_count:]

# 6. Save
train_ids.tofile("data/bin/train.bin")
val_ids.tofile("data/bin/val.bin")
```

## Example: Text → Tokens

**Input Text** (`corpus.txt`):
```
The quick brown fox jumps over the lazy dog.
Machine learning is transforming the world.
```

**Tokenization Process**:

```
Line 1: "The quick brown fox jumps over the lazy dog."
  Tokens: [1, 334, 3855, 288, 267, 2959, 354, 267, 12397, 8885, 2]
          [<s>, The, quick, brown, fox, jumps, over, the, lazy, dog, </s>]

Line 2: "Machine learning is transforming the world."
  Tokens: [1, 5234, 1234, 456, 7890, 267, 9876, 2]
          [<s>, Machine, learning, is, transforming, the, world, </s>]

Combined: [1, 334, 3855, ..., 2, 1, 5234, ..., 2]
```

**Binary Format**:

```
train.bin structure:
  Byte 0-1:   Token 0 (uint16)
  Byte 2-3:   Token 1 (uint16)
  Byte 4-5:   Token 2 (uint16)
  ...
  Byte N-2:N  Token N/2 (uint16)

Total size: 325,004,796 tokens × 2 bytes = ~650 MB
```

## Dataset Statistics

### Corpus Size

```
Raw Text:
  - Total files: 19
  - Total size: ~1.4 GB
  - Total lines: ~5.2M

Tokenized:
  - Total tokens: 361,116,440
  - Train tokens: 325,004,796 (90%)
  - Val tokens: 36,111,644 (10%)
```

## Usage

### Prepare Data

```bash
# Tokenize corpus
python data/prepare_data.py
```

**Output:**
```
Loading tokenizer from Tokenizer/BPE...
Vocab size: 32000
EOS ID: 2
Reading data/raw/merged_text/corpus.txt...
Total lines: 5,234,567
Tokenizing...
100%|████████████| 5.2M/5.2M [02:34<00:00]
Total tokens: 361,116,440
Train tokens: 325,004,796
Val tokens:   36,111,644
✅ Saved binary files to data/bin/
```

### Load in Training

```python
from train.dataloader import DataLoader

loader = DataLoader("data/bin", batch_size=16, block_size=512, split="train")
x, y = loader.get_batch(device="cuda")

# x: [16, 512] input tokens
# y: [16, 512] target tokens (shifted by 1)
```

## Memory-Mapped Loading

The binary files are loaded using `np.memmap` for efficiency:

```python
# Traditional loading (BAD)
data = np.fromfile("train.bin", dtype=np.uint16)  # Loads 650MB into RAM!

# Memory-mapped loading (GOOD)
data = np.memmap("train.bin", dtype=np.uint16, mode='r')  # OS handles paging
```

**Benefits:**
- **No RAM overhead**: File stays on disk
- **Fast random access**: OS caches hot pages
- **Scalable**: Works with TB-scale datasets

## References

- [The Pile: An 800GB Dataset](https://arxiv.org/abs/2101.00027)
- [Data Quality for Language Models](https://arxiv.org/abs/2201.06009)
- [Efficient Data Loading](https://pytorch.org/docs/stable/data.html)
