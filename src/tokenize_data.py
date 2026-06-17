from collections import defaultdict
import string
import pandas as pd
from datasets import load_dataset

def normalze_data(data):
    norm_data = ""
    for text in data:
        norm_data += text
        norm_data += "<|endoftext|>"
    norm_data = norm_data.replace(" ", "Ġ") 
    words = {}
    word = ""
    for char in norm_data:
        if char in string.ascii_letters:
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

# Get all pairs with their frequencies
def get_pairs_with_freqs(splits: dict, words: dict):
    pairs = defaultdict(int)

    for word, freq in words.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for c in range(len(split) - 1):
            pair = (split[c], split[c + 1])
            pairs[pair] += freq
    
    return pairs

# [m, e, r, g, e]
def merge_pair(a, b, splits: dict, words: dict):
    for word in words.keys():
        split = splits[word]
        if a and b in word and len(split) > 1:
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i+1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1
            splits[word] = split
    return splits

def add_online_data(data):
    copy_data = data
    prompts_data = pd.read_csv("hf://datasets/fka/prompts.chat/prompts.csv")
    fineweb_edu_fortified  = load_dataset("airtrain-ai/fineweb-edu-fortified", split="train", streaming=True)
    dataset_head = fineweb_edu_fortified.take(2)

    for entry in prompts_data["prompt"]:
        data.append(entry)
    
    for entry in fineweb_edu_fortified["text"]:
        data.append(entry)
    
    print(len(copy_data))
    return copy_data

# Tokenize data with Byte-Pair Encoding tokanization
def tokenize_data(data, vocab_size: int):

    data = add_online_data(data)

    norm_data, words = normalze_data(data)

    # Filter out characters from normalized data
    alphabet = sorted(list(set(norm_data)))

    # Add special token "end of text"
    base_vocab = ["<|pad|>", "<|endoftext|>"] + alphabet.copy()
    
    # Split each word into chars
    splits = defaultdict(list)

    # Create dict of chars for each word
    for word in words.keys():
        for c in word:
            splits[word].append(c)
    
    merges = defaultdict(tuple)

    # Run Splitting and merging loop until token size is reached
    while len(base_vocab) < vocab_size:
        pairs = get_pairs_with_freqs(splits, words)
        best_pair: tuple = ()
        highest_freq = None
        for pair, freq in pairs.items():
            if highest_freq is None or highest_freq < freq:
                best_pair = pair
                highest_freq = freq
        splits = merge_pair(best_pair[0], best_pair[1], splits, words)
        merges[best_pair] = best_pair[0] + best_pair[1]
        base_vocab.append(best_pair[0] + best_pair[1])
    
    ids_to_tokens = {}
    for i, token in enumerate(base_vocab):
        ids_to_tokens[i] = token

    tokens_to_ids = {}
    for i, token in enumerate(base_vocab):
        tokens_to_ids[token] = i
    
    return base_vocab, merges, ids_to_tokens, tokens_to_ids