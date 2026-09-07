import torch

from config import Config
from model import use_transformer
from tokenize_data import Tokenizer


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer()
    tokenizer.target_vocab_size = 300
    tokenizer.tokenize_data([
        {"text": "This is a small test corpus."},
        {"text": "Transformers learn next-token prediction."},
    ])
    return tokenizer


def test_tokenizer_builds_vocab_and_round_trips_text():
    tokenizer = build_tokenizer()

    assert tokenizer.tokens_to_ids([b"<|pad|>"]) == [0]
    assert tokenizer.tokens_to_ids([b"<|endoftext|>"]) == [1]

    encoded = tokenizer.encode_to_tensor(["This is a test."])
    decoded = tokenizer.decode_from_tensor(encoded)

    assert encoded.dtype == torch.long
    assert decoded == "This is a test.<|endoftext|>"


def test_transformer_forward_and_backward():
    tokenizer = build_tokenizer()
    config = Config.from_name("test", len(tokenizer.vocab))
    model = use_transformer(config)

    tokens = tokenizer.encode_to_tensor(["This is a test sequence."])
    source = tokens[:-1].unsqueeze(0)
    target = tokens[1:].unsqueeze(0)
    sequence_length = source.size(1)
    mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool))
    mask = mask.unsqueeze(0).unsqueeze(0)

    logits, loss = model(source, mask, target)
    loss.backward()

    assert logits.shape == (1, sequence_length, len(tokenizer.vocab))
    assert loss.ndim == 0
    assert any(parameter.grad is not None for parameter in model.parameters())