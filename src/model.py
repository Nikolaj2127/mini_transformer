from collections import defaultdict
from typing import Any, List
from tokenize_data import tokenize_data

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

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

## Tokenization
def encoderr(text, token_ids, merges):
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

    ids = [token_ids[t] for t in tokens]
    return torch.tensor(ids, dtype=torch.long)

def decoderr(pred_ids, token_ids: dict[int, str]):
    out: List[str] = []
    pred_ids_array = pred_ids.detach().cpu().flatten().tolist()

    for id in pred_ids_array:
        out.append(str(token_ids[id]))

    out_str = "".join(out)
    return out_str.replace("Ġ", " ")

## Embedding
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)

    def forward(self, x):
        return self.embedding(x)

## Positional Encoding
class PositionalEncoding(nn.Module):
    # PE(pos,2i) = sin(pos / (10000 ^ (2i/n_embd))
    # PE(pos,2i+1) = cos(pos / (10000 ^ (2i/n_embd))
    def __init__(self, n_embd, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, n_embd)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float) * (-math.log(10000) / n_embd))

        pe[:, 0::2 ] = torch.sin(pos * div)
        pe[:, 1::2 ] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.get_buffer("pe")
        x = x + pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

## Encoding
class Encoder(nn.Module):
    def __init__(self, n_embd: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(n_embd)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderBlock(nn.Module):
    def __init__(self, self_att_block, feed_forward_block, n_embd, dropout):
        super().__init__()
        self.self_att_block = self_att_block
        self.feed_forward_block = feed_forward_block
        self.res_conn = nn.ModuleList([ResidualConnection(n_embd, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Norm each x before calling the self_att_block.forward
        x = self.res_conn[0](x, lambda x: self.self_att_block(x, x, x, src_mask))
        x = self.res_conn[1](x, self.feed_forward_block)
        return x

## Multi head attention
class MultiHeadAttenentionBlock(nn.Module):
    def __init__(self, n_embd, h, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.h = h
        
        assert n_embd % h == 0, "n_embd is not devisible by h"

        self.dim_per_head = n_embd // h
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, dropout):
        dim_key = query.shape[-1]
        att_scores = (query @ key.transpose(-2, -1)) / math.sqrt(dim_key)
        if mask is not None:
            att_scores = att_scores.masked_fill(mask==0, -1e9)
        att_scores = att_scores.softmax(dim=-1)
        if dropout is not None:
            att_scores = dropout(att_scores)
        
        return (att_scores @ value), att_scores
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

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
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

## Feed forward
class FeedForward(nn.Module):
    def __init__(self, n_embd, n_blocks, dropout):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_blocks)
        self.linear2 = nn.Linear(n_blocks, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

## Layer normalization
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        standard_deviation = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (standard_deviation + self.eps) + self.bias

## Decoding
class DecoderBlock(nn.Module):
    def __init__(self, n_embd, self_att_block, cross_att_block, feed_forward_block, dropout) -> None:
        super().__init__()
        self.self_att_block = self_att_block
        self.cross_att_block = cross_att_block
        self.feed_forward_block = feed_forward_block
        self.res_conn = nn.ModuleList([ResidualConnection(n_embd, dropout) for _ in range(3)])
    
    def forward(self, x, enc_out, src_mask, trgt_mask):
        x = self.res_conn[0](x, lambda x: self.self_att_block(x, x, x, trgt_mask))
        x = self.res_conn[1](x, lambda x: self.cross_att_block(x, enc_out, enc_out, src_mask))
        x = self.res_conn[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, n_embd, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(n_embd)

    def forward(self, x, enc_out, src_mask, trgt_mask):
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embd, trgt_embd, src_pos, trgt_pos, proj_layer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embd = src_embd
        self.trgt_embd = trgt_embd
        self.src_pos = src_pos
        self.trgt_pos = trgt_pos
        self.proj_layer = proj_layer
    
    def encode(self, src, src_mask):
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, enc_out, src_mask, trgt, trgt_mask):
        trgt = self.trgt_embd(trgt)
        trgt = self.trgt_pos(trgt)
        return self.decoder(trgt, enc_out, src_mask, trgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)
    
    def forward(self, src_ids, src_mask, trgt_ids = None, trgt_mask = None):
        enc_out = self.encode(src_ids, src_mask)
        
        # encode source
        dec_out = self.decode(enc_out, src_mask, trgt_ids, trgt_mask)
        logits: torch.Tensor = self.project(dec_out)

        loss = None

        if trgt_ids is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), trgt_ids.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, prompt, token_to_ids, ids_to_token, merges, max_new_tokens, device):
        src_ids = encoderr(prompt, token_to_ids, merges).unsqueeze(0).to(device)
        trgt_ids = torch.tensor([[token_to_ids["<|endoftext|>"]]], device=device)

        for _ in range(max_new_tokens):
            logits, _ = self(src_ids, None, trgt_ids)
            next_logits = logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            trgt_ids = torch.cat([trgt_ids, next_id], dim=1)

            if next_id.item() == token_to_ids["<|endoftext|>"]:
                break
        
        return decoderr(trgt_ids, ids_to_token)

def use_transformer(vocab_size, n_embd, dropout, N, h, n_blocks):
    # Source and target token embedding
    src_tok_embd = InputEmbedding(vocab_size, n_embd)
    trgt_tok_embd = InputEmbedding(vocab_size, n_embd)

    # Source and target position embedding
    src_pos_embd = PositionalEncoding(n_embd, dropout)
    trgt_pos_embd = PositionalEncoding(n_embd, dropout)

    # Encoder blocks
    encoder_blocks = []

    for _ in range(N):
        enc_att_block = MultiHeadAttenentionBlock(n_embd, h, dropout)

        feed_forward_block = FeedForward(n_embd, n_blocks, dropout)

        enc_block = EncoderBlock(enc_att_block, feed_forward_block, n_embd, dropout)
        encoder_blocks.append(enc_block)

    # Decoder blocks
    decoder_blocks = []

    for _ in range(N):
        dec_self_att_block = MultiHeadAttenentionBlock(n_embd, h, dropout)
        dec_cross_att_block = MultiHeadAttenentionBlock(n_embd, h, dropout)

        feed_forward_block = FeedForward(n_embd, n_blocks, dropout)

        decoder_block = DecoderBlock(n_embd, dec_self_att_block, dec_cross_att_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(n_embd, nn.ModuleList(encoder_blocks))
    decoder = Decoder(n_embd, nn.ModuleList(decoder_blocks))

    # Create projection layer
    proj_layer = ProjectionLayer(n_embd, vocab_size)

    # Create transformer
    transformer = Transformer(encoder, decoder, src_tok_embd, trgt_tok_embd, src_pos_embd, trgt_pos_embd, proj_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer