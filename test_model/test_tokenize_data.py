from tokenize_data import Tokenizer


def test_normalize_data_preserves_word_boundaries():
    tokenizer = Tokenizer()

    assert tokenizer.normalize_data("hello world") == "helloĠworld"


def test_tokenizer_accepts_streaming_style_records():
    tokenizer = Tokenizer()
    tokenizer.target_vocab_size = 300

    tokenizer.tokenize_data([
        {"text": "alpha beta"},
        {"text": "beta gamma"},
    ])

    assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
    assert len(tokenizer.tokens_to_ids_map) == len(tokenizer.vocab)
