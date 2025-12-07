# Tokenizer Module

This module handles all tokenization tasks for the Mini-LLM project, converting raw text into numerical tokens that the model can process.

## Overview

The tokenizer uses **SentencePiece** with **Byte Pair Encoding (BPE)** to create a 32,000 token vocabulary. BPE is the same algorithm used by GPT-3, GPT-4, and LLaMA models.

## Directory Structure

```
Tokenizer/
├── BPE/                      # BPE tokenizer artifacts
│   ├── spm.model            # Trained SentencePiece model
│   ├── spm.vocab            # Vocabulary file
│   ├── tokenizer.json       # HuggingFace format
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── Unigram/                 # Unigram tokenizer (baseline)
│   └── ...
├── train_spm_bpe.py         # Train BPE tokenizer
├── train_spm_unigram.py     # Train Unigram tokenizer
└── convert_to_hf.py         # Convert to HuggingFace format
```

## How It Works

### 1. Training the Tokenizer

**Script**: `train_spm_bpe.py`

```python
import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="data/raw/merged_text/corpus.txt",
    model_prefix="Tokenizer/BPE/spm",
    vocab_size=32000,
    model_type="bpe",
    byte_fallback=True,  # Handles emojis, special chars
    character_coverage=1.0,
    user_defined_symbols=["<user>", "<assistant>", "<system>"]
)
```

**What happens:**
1. Reads raw text corpus
2. Learns byte-pair merges (e.g., "th" + "e" → "the")
3. Builds 32,000 most frequent tokens
4. Saves model to `spm.model`

### 2. Example: Tokenization Process

**Input Text:**
```
"Hello world! <user> write code </s>"
```

**Tokenization Steps:**

```
┌─────────────────────────────────────────┐
│ 1. Text Input                           │
│    "Hello world! <user> write code"     │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. BPE Segmentation                     │
│    ['H', 'ello', '▁world', '!',         │
│     '▁', '<user>', '▁write', '▁code']   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Token IDs                            │
│    [334, 3855, 288, 267, 2959,          │
│     354, 267, 12397]                    │
└─────────────────────────────────────────┘
```

**Key Features:**
- `▁` represents space (SentencePiece convention)
- Special tokens like `<user>` are preserved
- Byte fallback handles emojis: 🔥 → `<0xF0><0x9F><0x94><0xA5>`

### 3. Converting to HuggingFace Format

**Script**: `convert_to_hf.py`

```python
from transformers import LlamaTokenizerFast

tokenizer = LlamaTokenizerFast(vocab_file="Tokenizer/BPE/spm.model")
tokenizer.add_special_tokens({
    'bos_token': '<s>',
    'eos_token': '</s>',
    'unk_token': '<unk>',
    'pad_token': '<pad>'
})
tokenizer.save_pretrained("Tokenizer/BPE")
```

This creates `tokenizer.json` and config files compatible with HuggingFace Transformers.

## Usage

### Load Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Tokenizer/BPE")
```

### Encode Text

```python
text = "Hello world!"
ids = tokenizer.encode(text)
# Output: [1, 334, 3855, 288, 267, 2]
#         [<s>, H, ello, ▁world, !, </s>]
```

### Decode IDs

```python
decoded = tokenizer.decode(ids)
# Output: "<s> Hello world! </s>"

decoded = tokenizer.decode(ids, skip_special_tokens=True)
# Output: "Hello world!"
```

## BPE vs Unigram

| Feature | BPE | Unigram |
|---------|-----|---------|
| **Algorithm** | Merge frequent pairs | Probabilistic segmentation |
| **Emoji Handling** | ✅ Byte fallback | ❌ Creates `<unk>` |
| **URL Handling** | ✅ Clean splits | ⚠️ Unstable |
| **Used By** | GPT-3, GPT-4, LLaMA | BERT, T5 |
| **Recommendation** | ✅ **Primary** | Baseline only |

## Vocabulary Statistics

- **Total Tokens**: 32,000
- **Special Tokens**: 4 (`<s>`, `</s>`, `<unk>`, `<pad>`)
- **User-Defined**: 3 (`<user>`, `<assistant>`, `<system>`)
- **Coverage**: 100% (byte fallback ensures no `<unk>`)

## Performance

- **Compression Ratio**: ~3.5 bytes/token (English text)
- **Tokenization Speed**: ~1M tokens/second
- **Vocab Usage**: ~70% of tokens used in typical corpus

## References

- [SentencePiece Documentation](https://github.com/google/sentencepiece)
- [BPE Paper (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [Tokenizer Comparison Report](../tokenizer_report.md)
