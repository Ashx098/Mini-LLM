import sentencepiece as spm

spm.SentencePieceTrainer.Train(
    input="/home/aviinashh/projects/Mini-LLM/data/raw/merged_text/corpus.txt",
    model_prefix="/home/aviinashh/projects/Mini-LLM/Tokenizer/spm",
    vocab_size=32000,
    model_type="unigram",
    character_coverage=1.0,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    user_defined_symbols=["<user>", "<assistant>", "<system>"],
)

print("Tokenizer trained!")
# Model and vocab will be saved as spm.model and spm.vocab in the specified path