import torch
import torch.nn as nn
from model.transformer_block import TransformerBlock
from model.embedding import TokenEmbedding
from model.rmsnorm import RMSNorm
from model.rope import RoPE

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        # 1. Token Embeddings
        self.tok_embeddings = TokenEmbedding(args.vocab_size, args.dim)
        
        # 2. Dropout
        self.dropout = nn.Dropout(args.dropout)
        
        # 3. Transformer Blocks (The "Stack")
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
            
        # 4. Final Normalization
        self.norm = RMSNorm(args.dim)
        
        # 5. Output Head
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # 6. RoPE Generator
        # We pre-compute this here so we don't re-compute it every forward pass
        head_dim = args.dim // args.n_heads
        self.rope = RoPE(head_dim, args.max_seq_len)
        
        # 7. Weight Tying (Optional but recommended for small models)
        self.output.weight = self.tok_embeddings.emb.weight

    def forward(self, tokens, start_pos=0, use_cache=False, past_kv=None):
        # tokens: (Batch, Seq_Len)
        batch_size, seq_len = tokens.shape
        
        # 1. Embeddings
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        
        # 2. Get RoPE Frequencies
        # We slice the pre-computed frequencies based on current position
        freqs_cis = self.rope(start_pos + seq_len)
        # If inferencing (seq_len=1), we take the slice corresponding to current step
        if start_pos > 0:
            freqs_cis = freqs_cis[start_pos : start_pos + seq_len]
        
        # 3. Pass through Layers
        new_kvs = []
        for i, layer in enumerate(self.layers):
            # If using cache, pull the specific layer's past KV
            layer_past_kv = past_kv[i] if past_kv is not None else None
            
            h, layer_kv = layer(
                h, 
                freqs_cis=freqs_cis, 
                use_cache=use_cache, 
                past_kv=layer_past_kv
            )
            
            if use_cache:
                new_kvs.append(layer_kv)
                
        # 4. Final Norm
        h = self.norm(h)
        
        # 5. Output Projection (Logits)
        # We only care about the last token's prediction during inference usually, 
        # but for training we return everything.
        logits = self.output(h)
        
        return logits, new_kvs