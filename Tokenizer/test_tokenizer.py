from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(".")
print(tok.tokenize("Hello world! <user> write code </s>"))

text = "Hello world! <user> write code </s>"
ids = tok.encode(text)
print(ids)
print(tok.decode(ids))
print(tok.decode(ids, skip_special_tokens=True))