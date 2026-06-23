from collections import defaultdict
from itertools import islice
import string
from typing import List
from datasets import IterableDataset, load_dataset
import logging
logger = logging.getLogger(__name__)


class Tokenizer():
    def __init__(self) -> None:
        self.num_dataset_entries: int = 50
        self.target_vocab_size: int = 50
        self.TOK_DATA_ITER: int = 1
        self.vocab: List[str] = []
        self.special_tokens: List[str] = ["<|pad|>", "<|endoftext|>"]
        self.base_vocab: set[str] = set()
        self.words: dict[str, int] = {}
        self.merges: dict[tuple[str, str], str] = {}
        self.tokens_to_ids_map: dict[str, int] = {}
        self.ids_to_tokens_map: dict[int, str] = {}

    def fetch_online_data(self, dataset: IterableDataset) -> List[str]:
        logger.info("Fetching online data")
        copy_data = [entry["text"] for entry in islice(dataset, self.num_dataset_entries)]
        return copy_data

    def normalze_data(self, data: str) -> str:
        logger.info("Normalizing data")
        norm_data = data.replace(" ", "Ġ") 
        return norm_data

    # Get all pairs with their frequencies
    def get_pairs_with_freqs(self, splits: dict[str, List[str]], words: dict[str, int]) -> defaultdict[tuple[str, str], int]:
        logger.info("Getting pairs with freqs")
        pairs = defaultdict(int)

        for word, freq in words.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for c in range(len(split) - 1):
                pair = (split[c], split[c + 1])
                pairs[pair] += freq
        return pairs
    
    def update_base_vocab(self, norm_data: str) -> None:
        logger.info("Updating base vocabulary")
        self.base_vocab = self.base_vocab.union(set(norm_data))
    
    def update_words(self, norm_data: str) -> None:
        logger.info("Updating words and frequencies")
        word: str = ""
        for char in norm_data:
            if char in string.ascii_letters:
                word += char
            else:
                if word != "":
                    if word in self.words:
                        self.words[word] += 1
                    else:
                        self.words[word] = 1
                word = ""
                if char == "Ġ":
                    word += char
                else:
                    if char in self.words:
                        self.words[char] += 1
                    else:
                        self.words[char] = 1

    # [m, e, r, g, e]
    def merge_pair(self, a, b, splits: dict[str, List[str]], words: dict[str, int]) -> dict[str, List[str]]:
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

    def split_and_merge_data(self) -> None:
        logger.info("Splitting and merging data")
        # Split each word into chars
        splits: dict[str, List[str]] = defaultdict(list)

        # Create dict of chars for each word
        for word in self.words.keys():
            for c in word:
                splits[word].append(c)
        
        merges: dict[tuple[str, str], str] = defaultdict(tuple[str, str])
        self.vocab = sorted(list(self.base_vocab))

        # Run Splitting and merging loop until token size is reached
        while len(self.vocab) < self.target_vocab_size - len(self.special_tokens):
            pairs = self.get_pairs_with_freqs(splits, self.words)
            # If no pairs are found, we cannot merge anything else
            if not pairs:
                logger.error("Warning: Ran out of pairs to merge before reaching target vocabulary size.")
                break
            best_pair: tuple[str, str] = ("", "")
            highest_freq = None
            for pair, freq in pairs.items():
                if highest_freq is None or highest_freq < freq:
                    best_pair = pair
                    highest_freq = freq
            splits = self.merge_pair(best_pair[0], best_pair[1], splits, self.words)
            merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.append(best_pair[0] + best_pair[1])
        
        self.merges = merges
        self.vocab = self.special_tokens + self.vocab

    def update_token_id_dicts(self):
        logger.info("Updating token/id dictionaries")
        if self.vocab is not None:
            tokens_to_ids: dict[str, int] = {}
            ids_to_tokens: dict[int, str] = {}
            for i, token in enumerate(self.vocab):
                tokens_to_ids[token] = i
                ids_to_tokens[i] = token
            self.tokens_to_ids_map = tokens_to_ids
            self.ids_to_tokens_map = ids_to_tokens
        else:
            logger.error("No base_vocab.")
            return

    # Tokenize data with Byte-Pair Encoding tokanization
    def tokenize_data(self, dataset: IterableDataset):
        data_arr: List[str] = self.fetch_online_data(dataset)
        norm_data: str = ""
        for d in data_arr:
            norm_data += self.normalze_data(d)
        self.update_base_vocab(norm_data)
        self.update_words(norm_data)

        self.split_and_merge_data()

        self.update_token_id_dicts()
    
    """ Helper functions """

    def ids_to_tokens(self, token_ids: List[int]):
        logger.info("Converting ids to tokens")
        tokens = [self.ids_to_tokens_map[t] for t in token_ids]
        return tokens

    def tokens_to_ids(self, tokens: List[str]):
        logger.info("Converting tokens to ids")   
        ids = []
        for token in tokens:
            if token in self.tokens_to_ids_map:
                ids.append(self.tokens_to_ids_map[token])
            else:
                logger.error(f"Token {token} does not exist in base_vocab, skipping...")
        return ids
    
# text -> normalize: norm_text, words -> add words to wordlist -> perform splitting and merging -> encode vocab into ascii numbers for indexing