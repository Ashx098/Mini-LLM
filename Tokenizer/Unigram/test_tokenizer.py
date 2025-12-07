from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./Tokenizer/Unigram")


text1 = "Hello world! <user> write code </s>"
text2 = "myHTTPRequestHandler is calling process_payment_v2"
text3 = "methylphenidate hydrochloride dopamine reuptake modulation"
text4 = "hello 🔥🔥🔥💀💀"
text5 = "https://github.com/Avinash-MiniLLM?tab=repos"


print(text1)
print(text2)
print(text3)
print(text4)
print(text5)

print(tok.tokenize(text1))
print(tok.tokenize(text2))
print(tok.tokenize(text3))
print(tok.tokenize(text4))
print(tok.tokenize(text5))


ids1 = tok.encode(text1)
ids2 = tok.encode(text2)
ids3 = tok.encode(text3)
ids4 = tok.encode(text4)
ids5 = tok.encode(text5)

print(ids1)
print(tok.decode(ids1))
print(tok.decode(ids1, skip_special_tokens=True))

print(ids2)
print(tok.decode(ids2))
print(tok.decode(ids2, skip_special_tokens=True))

print(ids3)
print(tok.decode(ids3))
print(tok.decode(ids3, skip_special_tokens=True))

ids4 = tok.encode(text4)
print(ids4)
print(tok.decode(ids4))
print(tok.decode(ids4, skip_special_tokens=True))

ids5 = tok.encode(text5)
print(ids5)
print(tok.decode(ids5))
print(tok.decode(ids5, skip_special_tokens=True))