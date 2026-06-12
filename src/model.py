import torch

data = None

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

## Embedding

## Positional Encoding

## Self-Attention

## Feed-Forward NN (MLP)

## Predicting and Generating