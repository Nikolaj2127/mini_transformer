from collections import defaultdict
from itertools import islice
import string
from typing import List
import torch
from torch import Tensor
from datasets import IterableDataset, load_dataset
import logging
logger = logging.getLogger(__name__)


class Tokenizer():
    def __init__(self) -> None:
        self.num_dataset_entries: int = 500
        self.target_vocab_size: int = 2000
        self.TOK_DATA_ITER: int = 1
        self.vocab: List[bytes] = []
        self.special_tokens: List[bytes] = [b"<|pad|>", b"<|endoftext|>"]
        self.base_vocab: List[bytes] = [bytes([i]) for i in range(256)]
        self.words: dict[tuple[bytes, ...], int] = {}
        self.merges: dict[tuple[bytes, bytes], bytes] = {}
        self.tokens_to_ids_map: dict[bytes, int] = {}
        self.ids_to_tokens_map: dict[int, bytes] = {}

    def fetch_online_data(self, dataset: IterableDataset) -> List[str]:
        logger.info("Fetching online data")
        copy_data = [entry["text"] for entry in islice(dataset, self.num_dataset_entries)]
        return copy_data

    def normalze_data(self, data: str) -> str:
        logger.info("Normalizing data")
        norm_data = data.replace(" ", "Ġ") 
        return norm_data

    # Get all pairs with their frequencies
    def get_pairs_with_freqs(self, splits: dict[tuple[bytes, ...], List[bytes]], words: dict[tuple[bytes, ...], int]) -> defaultdict[tuple[bytes, bytes], int]:
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
    
    def update_words(self, norm_data: str) -> None:
        logger.info("Updating words and frequencies")

        word: str = ""
        for char in norm_data:
            if char in string.ascii_letters:
                word += char
            else:
                if word != "":
                    b_word = tuple(bytes([b]) for b in word.encode("utf-8"))
                    if b_word in self.words:
                        self.words[b_word] += 1
                    else:
                        self.words[b_word] = 1
                word = ""
                if char == "Ġ":
                    word += char
                else:
                    b_char = tuple(bytes([b]) for b in char.encode("utf-8"))
                    if b_char in self.words:
                        self.words[b_char] += 1
                    else:
                        self.words[b_char] = 1

    # [m, e, r, g, e]
    def merge_pair(self, a: bytes, b: bytes, splits: dict[tuple[bytes, ...], List[bytes]], words: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, ...], List[bytes]]:
        for word in words.keys():
            split = splits[word]
            if a in split and b in split and len(split) > 1:
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
        splits: dict[tuple[bytes, ...], List[bytes]] = defaultdict(list)

        # Create dict of chars for each word
        for word in self.words.keys():
            splits[word] = list(word)
        
        merges: dict[tuple[bytes, bytes], bytes] = defaultdict(tuple[bytes, bytes])
        self.vocab = sorted(list(self.base_vocab))

        # Run Splitting and merging loop until token size is reached
        while len(self.vocab) < self.target_vocab_size - len(self.special_tokens):
            pairs = self.get_pairs_with_freqs(splits, self.words)
            # If no pairs are found, we cannot merge anything else
            if not pairs:
                logger.error("Warning: Ran out of pairs to merge before reaching target vocabulary size.")
                break
            best_pair: tuple[bytes, bytes] = (b"", b"")
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
            tokens_to_ids: dict[bytes, int] = {}
            ids_to_tokens: dict[int, bytes] = {}
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
        self.update_words(norm_data)

        self.split_and_merge_data()

        self.update_token_id_dicts()
    
    """ Helper functions """

    def ids_to_tokens(self, token_ids: List[int]):
        #logger.info("Converting ids to tokens")
        tokens = [self.ids_to_tokens_map[t] for t in token_ids]
        return tokens

    def tokens_to_ids(self, tokens: List[bytes]):
        #logger.info("Converting tokens to ids")   
        ids = []
        for token in tokens:
            if token in self.tokens_to_ids_map:
                ids.append(self.tokens_to_ids_map[token])
            else:
                logger.error(f"Token {token} does not exist in base_vocab, skipping...")
        return ids
    
    def encode_to_tensor(self, text_arr: List[str]):
        all_tokens = []
        for text in text_arr:
            text = text.replace(" ", "Ġ")
            byte_data = text.encode("utf-8")
            tokens = [bytes([b]) for b in byte_data]

            for (a, b), merged in self.merges.items():
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
            
            tokens.append(b"<|endoftext|>")
        
            all_tokens += tokens

        ids = self.tokens_to_ids(all_tokens)
        return torch.tensor(ids, dtype=torch.long)

    def decode_from_tensor(self, pred_ids: Tensor):
        pred_ids_array = pred_ids.detach().cpu().flatten().tolist()

        out: List[bytes] = self.ids_to_tokens(pred_ids_array)

        out_bytes = b"".join(out)
        out_str = out_bytes.decode("utf-8", errors="replace")
        return out_str.replace("Ġ", " ")
    
# text -> normalize: norm_text, words -> add words to wordlist -> perform splitting and merging -> encode vocab into ascii numbers for indexing