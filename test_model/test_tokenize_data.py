
from tokenize_data import *

class TestTokenizeData():
    def test_tokenize_data(self):
        pass

    def test_tokens_to_ids_single(self):
        tokenizer = Tokenizer()

        tokenizer.update_words(["ab"])
        tokenizer.update_base_vocab("ad")

        ids_a = tokenizer.tokens_to_ids(["a"])
        ids_d = tokenizer.tokens_to_ids(["d"])
        ids_ab = tokenizer.tokens_to_ids(["ab"])

        print(ids_a)
        print(ids_d)
        print(ids_ab)

        assert ids_a == [9702]
        assert ids_d == [1003]
        assert ids_ab == [97029802]
    
    def test_ids_to_tokens(self):
        tokenizer = Tokenizer()

        tokens_a = tokenizer.ids_to_tokens([9702])
        tokens_d = tokenizer.ids_to_tokens([1003])
        tokens_ab = tokenizer.ids_to_tokens([97029802])

        print(tokens_a)
        print(tokens_d)
        print(tokens_ab)

        assert tokens_a == ["a"]
        assert tokens_d == ["d"]
        assert tokens_ab == ["ab"]
    
    def test_tokenize_all(self):
        tokenizer = Tokenizer()

        data = "This is a test lol."

        tokenizer.tokenize_data(data)

        ids = tokenizer.tokens_to_ids(["is", "a"])
        tokens = tokenizer.ids_to_tokens(ids)

        print(ids, tokens)

        assert tokens == ["is", "a"]
    
    def test_diff_token(self):
        tokenizer = Tokenizer()

        tokenizer.tokenize_data()

        print(tokenizer.base_vocab)
        print(tokenizer.words)
