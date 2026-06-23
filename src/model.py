from __future__ import annotations
from collections import defaultdict
from typing import Any, List, Callable
from tokenize_data import *

import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

class Config():
    def __init__(self, vocab_size: int, n_embd: int = 256, dropout: float = 0.1, N: int = 6, h: int = 8):
        self.n_embd: int = n_embd
        self.vocab_size: int = vocab_size
        self.dropout: float = dropout
        self.N: int = N
        self.h: int = h
        self.n_blocks: int = 4 * n_embd

vocab_size = 50
block_size = 64
n_embd = 64
n_blocks = 64

"""
Tokenize data

Create input embedding Tensor
Create output embedding Tensor

Create input positional embedding Tensor
Create output positional embedding Tensor

for each (N)
    Create MultiHeadAttentionBlock
    Create FeedForwardBlock
    Create EncoderBlock
    add EncoderBlock to list of encoder blocks

for each (N)
    Create MultiHeadAttentionBlock for self attention
    Create MultiHeadAttentionBlock for cross attention
    Create FeedForwardBlock
    Create DecoderBlock
    add DecoderBlock to list of decoder blocks

Create Encoder
Create Decoder

Create Projection Layer

Create Transformer

Call model.encode - (msg, mask)
-> Call input token embedding - (msg)
-> Call input positional embedding - (msg)
-> Call encoder with - (msg, mask)
    for each layer
        -> Call encoder block - (msg, mask)
            -> sublayer - call multi head attention - (msg, msg, msg, mask)
                -> get attention scores and weights (predictions)
                -> return output tensor
            -> Call residual connection - (msg, sublayer)
            -> output tensor - Call residual connection (output tensor, feed forward block)
            -> return output tensor
    -> Normalize final output
-> Call decode
-> Call project
-> Get argmax of predictions
-> decode into words
-> output words
"""

## Embedding
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

## Positional Encoding
class PositionalEncoding(nn.Module):
    # PE(pos,2i) = sin(pos / (10000 ^ (2i/n_embd))
    # PE(pos,2i+1) = cos(pos / (10000 ^ (2i/n_embd))
    def __init__(self, n_embd: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe: Tensor = torch.zeros(max_len, n_embd)
        pos: Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div: Tensor = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float) * (-math.log(10000) / n_embd))

        pe[:, 0::2 ] = torch.sin(pos * div)
        pe[:, 1::2 ] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        pe: Tensor = self.get_buffer("pe")
        x = x + pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

## Encoding
class Decoder(nn.Module):
    def __init__(self, n_embd: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(n_embd)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_att_block: MultiHeadAttenentionBlock, feed_forward_block: FeedForward, n_embd: int, dropout: float):
        super().__init__()
        self.self_att_block = self_att_block
        self.feed_forward_block = feed_forward_block
        self.res_conn = nn.ModuleList([ResidualConnection(n_embd, dropout) for _ in range(2)])

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        # Norm each x before calling the self_att_block.forward
        x = self.res_conn[0](x, lambda x: self.self_att_block(x, x, x, src_mask))
        x = self.res_conn[1](x, self.feed_forward_block)
        return x

## Multi head attention
class MultiHeadAttenentionBlock(nn.Module):
    def __init__(self, n_embd: int, h: int, dropout: float):
        super().__init__()
        self.n_embd = n_embd
        self.h = h
        
        assert n_embd % h == 0, "n_embd is not devisible by h"

        self.dim_per_head: int = n_embd // h
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor, dropout: nn.Dropout) -> tuple[Tensor, Tensor]:
        dim_key: int = query.shape[-1]
        att_scores: Tensor = (query @ key.transpose(-2, -1)) / math.sqrt(dim_key)
        if mask is not None:
            att_scores = att_scores.masked_fill(~mask, -1e9)
        att_scores = att_scores.softmax(dim=-1)
        if dropout is not None:
            att_scores = dropout(att_scores)
        
        return (att_scores @ value), att_scores
        
    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor) -> Tensor:
        query: Tensor = self.query(query)
        key: Tensor = self.key(key)
        value: Tensor = self.value(value)

        query = query.view(query.shape[0], query.shape[1], self.h, self.dim_per_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dim_per_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dim_per_head).transpose(1, 2)

        x, self.att_scores = MultiHeadAttenentionBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.dim_per_head)

        return self.output(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
    
    def forward(self, x: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))

## Feed forward
class FeedForward(nn.Module):
    def __init__(self, n_embd: int, n_blocks: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_blocks)
        self.linear2 = nn.Linear(n_blocks, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

## Layer normalization
class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x: Tensor) -> Tensor:
        mean: Tensor = x.mean(dim=-1, keepdim=True)
        standard_deviation: Tensor = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (standard_deviation + self.eps) + self.bias

class ProjectionLayer(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, decoder: Decoder, src_embd: InputEmbedding, src_pos: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.decoder = decoder
        self.src_embd = src_embd
        self.src_pos = src_pos
        self.proj_layer = proj_layer
    
    def project(self, x: Tensor) -> Tensor:
        return self.proj_layer(x)
    
    def forward(self, src_ids: Tensor, src_mask: Tensor, trgt_ids: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = self.src_embd(src_ids)
        x = self.src_pos(x)
        
        # encode source
        dec_out: Tensor = self.decoder(x, src_mask)
        logits: Tensor = self.project(dec_out)

        loss: Tensor | None = None

        if trgt_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), trgt_ids.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, model: Transformer, prompt: str, max_new_tokens: int, device: str, temp: int, tokenizer: Tokenizer) -> str:
        model.eval()

        src_ids: Tensor = tokenizer.encode_to_tensor([prompt]).unsqueeze(0).to(device)
        eos_id: int = tokenizer.tokens_to_ids([b"<|endoftext|>"])[0]

        if src_ids[0, -1].item() == eos_id:
            src_ids = src_ids[:, :-1]

        for _ in range(max_new_tokens):
            T: int = src_ids.size(1)
            causal: Tensor = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

            logits, _ = model(src_ids, causal)
            next_logits = logits[:, -1, :]

            probs: Tensor = torch.softmax(next_logits / temp, dim=-1)
            next_id: Tensor = torch.multinomial(probs, num_samples=1)

            if next_id.item() == eos_id:
                break
                
            src_ids: Tensor = torch.cat([src_ids, next_id], dim=1)

        return tokenizer.decode_from_tensor(src_ids)

def use_transformer(cfg: Config) -> Transformer:
    vocab_size: int = cfg.vocab_size
    n_embd: int = cfg.n_embd
    dropout: float = cfg.dropout
    N: int  = cfg.N
    h: int  = cfg.h
    n_blocks: int = cfg.n_blocks
    # Source and target token embedding
    src_tok_embd = InputEmbedding(vocab_size, n_embd)

    # Source and target position embedding
    src_pos_embd = PositionalEncoding(n_embd, dropout)

    # Decoder blocks
    decoder_blocks = []

    for _ in range(N):
        dec_self_att_block = MultiHeadAttenentionBlock(n_embd, h, dropout)

        feed_forward_block = FeedForward(n_embd, n_blocks, dropout)

        decoder_block = DecoderBlock(dec_self_att_block, feed_forward_block, n_embd, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    decoder = Decoder(n_embd, nn.ModuleList(decoder_blocks))

    # Create projection layer
    proj_layer = ProjectionLayer(n_embd, vocab_size)

    # Create transformer
    transformer = Transformer(decoder, src_tok_embd, src_pos_embd, proj_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer