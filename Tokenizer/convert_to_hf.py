from transformers import LlamaTokenizerFast

# Load the raw spm model
tokenizer = LlamaTokenizerFast(vocab_file="/home/aviinashh/projects/Mini-LLM/Tokenizer/BPE/spm.model")

# Add your special tokens manually to the HF config part
tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "additional_special_tokens": ["<user>", "<assistant>", "<system>"]
})

# Save the json version
tokenizer.save_pretrained("Tokenizer/")

print("Converted to tokenizer.json successfully!")