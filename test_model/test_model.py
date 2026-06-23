from collections import defaultdict

from src.model import *
from tokenize_data import *

def test_data():
    return [
                "This is the Hugging Face Course.",
                "This chapter is about tokenization.",
                "This section shows several tokenizer algorithms.",
                "Hopefully, you will be able to understand how they are trained and generate tokens.",
            ]

class TestModel:
    def test_normalization(self):
        tokenizer = Tokenizer()
        sample_text = test_data()
        norm_data, words = tokenizer.normalze_data(sample_text)
        assert norm_data == "ThisĠisĠtheĠHuggingĠFaceĠCourse.ThisĠchapterĠisĠaboutĠtokenization.ThisĠsectionĠshowsĠseveralĠtokenizerĠalgorithms.Hopefully,ĠyouĠwillĠbeĠableĠtoĠunderstandĠhowĠtheyĠareĠtrainedĠandĠgenerateĠtokens."

        assert words == {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1,
            'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1,
            'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1,
            'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1}

    def test_tokenize_data(self):
        tokenizer = Tokenizer()
        sample_text = test_data()
        vocab, merges = tokenizer.tokenize_data(sample_text, 50)

        assert merges == {('Ġ', 't'): 'Ġt', ('i', 's'): 'is', ('e', 'r'): 'er', ('Ġ', 'a'): 'Ġa', ('Ġt', 'o'): 'Ġto', ('e', 'n'): 'en',
            ('T', 'h'): 'Th', ('Th', 'is'): 'This', ('o', 'u'): 'ou', ('s', 'e'): 'se', ('Ġto', 'k'): 'Ġtok',
            ('Ġtok', 'en'): 'Ġtoken', ('n', 'd'): 'nd', ('Ġ', 'is'): 'Ġis', ('Ġt', 'h'): 'Ġth', ('Ġth', 'e'): 'Ġthe',
            ('i', 'n'): 'in', ('Ġa', 'b'): 'Ġab', ('Ġtoken', 'i'): 'Ġtokeni'}
        
        assert vocab == ['<|endoftext|>', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o',
            'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'Ġ', 'Ġt', 'is', 'er', 'Ġa', 'Ġto', 'en', 'Th', 'This', 'ou', 'se',
            'Ġtok', 'Ġtoken', 'nd', 'Ġis', 'Ġth', 'Ġthe', 'in', 'Ġab', 'Ġtokeni']
    
    def test_tokens_to_ids(self):
        tokenizer = Tokenizer()

        ids = tokenizer.tokenize_data("a")

        print(ids)

        assert ids == [9702]