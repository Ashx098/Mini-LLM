---
language:
- en
license: mit
tags:
- llm
- decoder-only
- transformer
- from-scratch
- research
- educational
- 80m
- pytorch
- pretraining
- custom-architecture
pipeline_tag: text-generation
inference:
  parameters:
    temperature: 0.7
    top_p: 0.95
---

# 🧠 Mini-LLM — 80M Parameter Transformer (Pretrained From Scratch)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Model Size](https://img.shields.io/badge/params-80M-blue.svg)]()

**Mini-LLM** is an 80M parameter decoder-only transformer trained **fully from scratch** using a custom tokenizer, custom architecture, and custom training loop.  
It is designed as an educational + research-friendly minimal LLM that demonstrates how modern LLM components are built end-to-end.

---

## ✨ Key Features

- **80M parameters** — compact but fully functional LLM  
- **Trained from scratch** (no borrowed checkpoints)  
- Custom **Byte-Level BPE tokenizer (32k vocab)**  
- Modern architecture components:
  - RoPE (Rotary Position Embeddings)
  - RMSNorm
  - SwiGLU FeedForward layer
  - FlashAttention (via PyTorch SDPA)
  - GQA-ready Attention implementation
- **2B tokens** mixed corpus (FineWeb + WikiText + Wikipedia)
- Training logs, checkpoints, plots all included for transparency
- Released under a permissive license for research & learning

---

## 📐 Model Architecture

| Component | Value |
|----------|-------|
| Type | Decoder-only transformer |
| Parameters | ~80M |
| Layers | 16 |
| Embedding dim | 384 |
| Attention heads | 6 |
| KV Heads | 6 |
| MLP Hidden Dim | 1536 (SwiGLU) |
| Max sequence length | 2048 |
| Norm | RMSNorm |
| Positional Encoding | RoPE |
| Tokenizer | SentencePiece BPE (32k vocab, byte fallback) |

---

## 📦 Files in This Repo

- `checkpoints/` → Pretrained model state_dict + optimizer
- `safetensors/` → Final consolidated .safetensors file
- `logs/` → Training logs in JSONL
- `plots/` → Train/val loss curves
- `tokenizer.json` → HF-compatible tokenizer
- `spm.model` → SentencePiece model

---

## 🧪 Quick Usage (HF Transformers)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Ashx098/Mini-LLM", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("Ashx098/Mini-LLM")

prompt = "Hello, how are you?"
inputs = tok(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=50)
print(tok.decode(outputs[0], skip_special_tokens=True))
```

## 🚀 Training Details

### Optimizer
- **AdamW** (β1=0.9, β2=0.95, weight decay=0.1)
- **Learning rate**: 6e-4 (cosine annealing + warmup)

### Batch ⨉ Sequence
- **Global batch size** = 32
- **Sequence length** = 2048
- **Gradient accumulation** = 8

### Hardware
- Trained on 1× NVIDIA A100 80GB

## 📊 Training Curve
<p align="center"> <img src="https://huggingface.co/Ashx098/Mini-LLM/resolve/main/phase-1-pretraining/plots/loss_curve.png" width="500"> </p>

Final loss reached: ~3.25

## 💬 Example Outputs

**Prompt**: "Hello, how are you"
**Output**: "Hello, how are you?"

**Prompt**: "Python is a programming language that"
**Output**: "Python is a programming language that allows the history..."

## ⚠️ Limitations
- Small model → limited reasoning, hallucination likely
- Not instruction-tuned
- Not suitable for production usage
- Best viewed as a learning + research artifact

## 📜 License
MIT License — free for research, modification, and further training.

## 🙌 Credits
Developed by **Avinash Mynampati**  
Built from scratch using PyTorch + custom training pipeline.

### Want to fine-tune or extend it?
You can:
- Train further with your own dataset
- Add LoRA adapters
- Use it to learn attention, RoPE, SwiGLU, etc.
- Build a tiny instruction-tuned version (coming soon!)

## 📬 Contact
For questions or collaborations:
- **GitHub**: [Ashx098](https://github.com/Ashx098)
- **LinkedIn**: [Avinash Mynampati](https://linkedin.com/in/avinash-mynampati)
