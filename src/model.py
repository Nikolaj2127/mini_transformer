import math

import torch
import torch.nn as nn

data = None
n_embd = 64
n_blocks = 64

## Tokenization
# Normalization
def normalze_data(data):
    norm_data = ""
    for text in data:
        norm_data += text
    norm_data = norm_data.replace(" ", "Ġ") 
    words = {}
    word = ""
    for i, char in enumerate(norm_data):
        if char not in "Ġ.,;:!?#$%&'()*+-/0123456789<=>@[\]^_`{|}~":
            word += char
        else:
            if word != "":
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
            word = ""
            if char == "Ġ":
                word += char
            else:
                if char in words:
                    words[char] += 1
                else:
                    words[char] = 1
    return norm_data, words

base_vocab = sorted(list(set(data)))

## Encoding
class EncoderBlock(nn.Module):
    def __init__(self, self_att_block, feed_forward_block, features, dropout):
        super().__init__()
        self.self_att_block = self_att_block
        self.feed_forward_block = feed_forward_block
        self.res_conn = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.res_conn[0](x, lambda x: self.self_att_block(x, x, x, src_mask))
        x = self.res_conn[1](x, self.feed_forward_block)

## Embedding
class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, n_embd):
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
        pos = torch.arange(0, max_len).unsqueeze(1)

        pe[:, 0::2 ] = torch.sin(pos / (10000 ** (torch.arange(0, n_embd, 2))))
        pe[:, 1::2 ] = torch.cos(pos / (10000 ** (torch.arange(0, n_embd, 2))))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(features)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


## Self-Attention

## Feed-Forward NN (MLP)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()


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

## Feed forward

class FeedForward(nn.Module):
    def __init__(self, n_embd, n_blocks, dropout):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, n_blocks)
        self.linear2 = nn.Linear(n_blocks, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
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
        att_scores = (query @ key.transpose(-2, 1)) / math.sqrt(dim_key)
        if mask is not None:
            att_scores.masked_fill(mask==0, -1e9)
        att_scores = att_scores.softmax(dim=-1)
        if dropout is not None:
            att_scores = dropout(att_scores)
        
        return (att_scores @ value), value
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        query = query.view(query.shape[0], query.shape[1], self.h, self.dim_per_head).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dim_per_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dim_per_head).transpose(1, 2)

        x, att_scores = MultiHeadAttenentionBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.dim_per_head)

        return self.output(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, n_embd, vocab_size):
        super().__init__()
        self.proj = nn.Linear(n_embd, vocab_size)

    def forward(self, x):
        return self.proj(x)

class Transformer(nn.Module):
    pass

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

    # Create encoder and decoder

    # Create projection layer
    projection_layer = ProjectionLayer(n_embd, vocab_size)

    # Create transformer

    return