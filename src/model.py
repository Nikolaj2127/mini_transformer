from collections import defaultdict
from typing import Any
from tokenize_data import tokenize_data

import torch
import torch.nn as nn

data = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
        ]
vocab_size = 50
num_embd = 64
block_size = 64

## Tokenization
vocab, merges = tokenize_data(data, 50)

token_to_id = {token: i for i, token in enumerate(vocab)}

def encode(text, token_to_id, merges):
    text = text.replace(" ", "Ġ")
    tokens = list(text)

    for (a, b), merged in merges.items():
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    ids = [token_to_id[t] for t in tokens]
    return torch.tensor(ids, dtype=torch.long)

x = encode("This is the Hugging Face Course.", token_to_id, merges).unsqueeze(0)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, num_embd)
        self.pos_emb = nn.Embedding(block_size, num_embd)

    def forward(self, idx: torch.Tensor):
        # Batch/ Time / Channel Tensors
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

## Embedding

## Positional Encoding

## Self-Attention

## Feed-Forward NN (MLP)

## Predicting and Generating

# TODOS
# TODO: Add proper encoding for text tokenization
# TODO: Save tokens to seperate file for better performance