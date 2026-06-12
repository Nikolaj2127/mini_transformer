from collections import defaultdict

from src.model import *

class TestModel:
    def test_normalization(self):
        corpus = [
            "This is the Hugging Face Course.",
            "This chapter is about tokenization.",
            "This section shows several tokenizer algorithms.",
            "Hopefully, you will be able to understand how they are trained and generate tokens.",
        ]

        norm_data, words =normalze_data(corpus)
        assert norm_data == "ThisĠisĠtheĠHuggingĠFaceĠCourse.ThisĠchapterĠisĠaboutĠtokenization.ThisĠsectionĠshowsĠseveralĠtokenizerĠalgorithms.Hopefully,ĠyouĠwillĠbeĠableĠtoĠunderstandĠhowĠtheyĠareĠtrainedĠandĠgenerateĠtokens."

        assert words == {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1, 'ĠFace': 1, 'ĠCourse': 1, '.': 4, 'Ġchapter': 1,
            'Ġabout': 1, 'Ġtokenization': 1, 'Ġsection': 1, 'Ġshows': 1, 'Ġseveral': 1, 'Ġtokenizer': 1, 'Ġalgorithms': 1,
            'Hopefully': 1, ',': 1, 'Ġyou': 1, 'Ġwill': 1, 'Ġbe': 1, 'Ġable': 1, 'Ġto': 1, 'Ġunderstand': 1, 'Ġhow': 1,
            'Ġthey': 1, 'Ġare': 1, 'Ġtrained': 1, 'Ġand': 1, 'Ġgenerate': 1, 'Ġtokens': 1}

        