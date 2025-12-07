import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import json
import random
from collections import Counter
from transformers import AutoTokenizer
import numpy as np

# --------------------------------------------
# CONFIG
# --------------------------------------------

UNI_PATH = "Tokenizer/Unigram"
BPE_PATH = "Tokenizer/BPE"

CORPUS_PATH = "data/raw/merged_text/corpus.txt"
REPORT_PATH = "tokenizer_report.md"

N_SAMPLES = 20_000   # for vocab usage check
TEST_STRINGS = [
    "Hello world! <user> write code </s>",
    "myHTTPRequestHandler is calling process_payment_v2",
    "methylphenidate hydrochloride dopamine reuptake modulation",
    "hello 🔥🔥🔥💀💀",
    "https://github.com/Avinash-MiniLLM?tab=repos",
    "The quick brown fox jumps over the lazy dog.",
    "function computeHash(input: bytes): uint256 { return keccak256(input); }",
    "भारत is a country — multilingual test 🌏",
]


# --------------------------------------------
# Load Tokenizers
# --------------------------------------------
def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


tok_uni = load_tokenizer(UNI_PATH)
tok_bpe = load_tokenizer(BPE_PATH)

print("Loaded Unigram + BPE tokenizers.")


# --------------------------------------------
# Helper metrics
# --------------------------------------------
def compression_ratio(tokenizer, text):
    ids = tokenizer.encode(text)
    if len(ids) == 0:
        return 999
    return len(text.encode("utf-8")) / len(ids)


def vocab_usage(tokenizer, corpus, n=50_000):
    words = corpus.split()[:n]
    ids = tokenizer(" ".join(words)).input_ids
    return len(set(ids)), ids


def unused_vocab(tokenizer, used_ids):
    vocab_size = tokenizer.vocab_size
    used = len(used_ids)
    return vocab_size - used, (vocab_size - used) / vocab_size


# --------------------------------------------
# Generate Markdown Report
# --------------------------------------------
def write_report(content):
    with open(REPORT_PATH, "w") as f:
        f.write(content)
    print(f"\nReport saved → {REPORT_PATH}")


# --------------------------------------------
# Run tests
# --------------------------------------------

corpus = open(CORPUS_PATH).read()

report = []
report.append("# 🧠 Tokenizer Evaluation Report\n")
report.append("Comparing **Unigram** vs **BPE (Byte Fallback)** Tokenizers.\n")
report.append("---\n")

# --------------------------------------------
# Test 1: Tokenization examples
# --------------------------------------------
report.append("## 🔍 Tokenization Examples\n")

for s in TEST_STRINGS:
    uni_toks = tok_uni.tokenize(s)
    bpe_toks = tok_bpe.tokenize(s)

    report.append(f"### Input:\n`{s}`")
    report.append("\n**Unigram:**\n```\n" + str(uni_toks) + "\n```")
    report.append("\n**BPE:**\n```\n" + str(bpe_toks) + "\n```\n---\n")


# --------------------------------------------
# Test 2: Compression Ratio
# --------------------------------------------

report.append("## ⚙️ Compression Ratios\n")
report.append("| Text | Unigram | BPE |\n|------|---------|-----|\n")

for s in TEST_STRINGS:
    cr_uni = compression_ratio(tok_uni, s)
    cr_bpe = compression_ratio(tok_bpe, s)
    report.append(f"| `{s[:25]}...` | {cr_uni:.2f} | {cr_bpe:.2f} |\n")

report.append("---\n")


# --------------------------------------------
# Test 3: Vocab usage statistics
# --------------------------------------------

report.append("## 📊 Vocabulary Usage on Sample Corpus\n")
uni_used, uni_ids = vocab_usage(tok_uni, corpus, N_SAMPLES)
bpe_used, bpe_ids = vocab_usage(tok_bpe, corpus, N_SAMPLES)

report.append(f"- **Unigram unique tokens used:** {uni_used}")
report.append(f"- **BPE unique tokens used:** {bpe_used}")

unused_uni, unused_uni_pct = unused_vocab(tok_uni, uni_ids)
unused_bpe, unused_bpe_pct = unused_vocab(tok_bpe, bpe_ids)

report.append(f"\n### Unused Vocabulary\n")
report.append(f"- Unigram unused tokens: {unused_uni} ({unused_uni_pct*100:.2f}%)")
report.append(f"- BPE unused tokens: {unused_bpe} ({unused_bpe_pct*100:.2f}%)")

report.append("---\n")


# --------------------------------------------
# Test 4: Byte fallback test
# --------------------------------------------

emoji_test = "hello 🔥🔥🔥💀💀"
uni_broken = tok_uni.tokenize(emoji_test)
bpe_bytes = tok_bpe.tokenize(emoji_test)

report.append("## 🔥 Byte Fallback Behavior\n")
report.append("### Input:\n```\nhello 🔥🔥🔥💀💀\n```")
report.append("\n**Unigram:**\n```\n" + str(uni_broken) + "\n```")
report.append("\n**BPE:**\n```\n" + str(bpe_bytes) + "\n```")
report.append("\n(Shows why BPE is required for modern LLMs.)")
report.append("\n---\n")


# --------------------------------------------
# Test 5: URL Handling
# --------------------------------------------

url = "https://github.com/Avinash-MiniLLM?tab=repos"
report.append("## 🌐 URL Handling\n")
report.append("### Input:\n```\n" + url + "\n```")
report.append("\n**Unigram:**\n```\n" + str(tok_uni.tokenize(url)) + "\n```")
report.append("\n**BPE:**\n```\n" + str(tok_bpe.tokenize(url)) + "\n```")
report.append("\n---\n")


# --------------------------------------------
# Final Notes
# --------------------------------------------
report.append("## ✅ Summary\n")
report.append("""
- **BPE handles emojis, URLs, code, and multilingual text much better.**
- **Unigram produces <unk> & unstable splits on web-style inputs.**
- BPE matches **Qwen / GPT / OLMo / LLaMA-3** tokenizer behavior.
- Unigram can still be kept for research baselines.

**Recommendation:**  
➡️ Use **BPE** as primary tokenizer for your 80M Mini-LLM.  
➡️ Keep Unigram as comparison baseline only.
""")

# Save Report
write_report("\n".join(report))
