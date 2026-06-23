from torch.optim.optimizer import StateDict
from typing import Generator
from torch import dropout
from pathlib import Path
import os
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

from model import *
from tokenize_data import *

def main():

    configure_logging()

    logger.info("Starting transformer")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Getting dataset")
    dataset: IterableDataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", filters=[("language", "==", "en")], split="train", streaming=True)

    logger.info("Initializing Tokenizer")
    tokenizer = Tokenizer()
    cfg = Config(len(tokenizer.vocab))

    logger.info("Tokenizing data")
    tokenizer.tokenize_data(dataset=dataset)

    logger.info("Initializing transformer")

    model: Transformer = use_transformer(cfg)

    logger.info("Training model")
    train_model(model, device, tokenizer, dataset)

    logger.info("Saving model")
    saved_model: Transformer = load(cfg)

    prompt: str = "The history of the Roman Empire begins with"

    logger.info("Generating output")
    out: str = model.generate(saved_model, prompt, max_new_tokens=50, device=device, temp=1, tokenizer=tokenizer)

    print(out)

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("mini_transformer.log"), logging.StreamHandler()])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

def save(model: Transformer, file_name: str = "model.pth") -> None:
        model_folder_path: str = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name: str = os.path.join(model_folder_path, file_name)
        torch.save(model.state_dict(), file_name)

def load(cfg: Config, file_name: str = "model.pth") -> Transformer:
    model_folder_path: str = "./model"
    file_name: str = os.path.join(model_folder_path, file_name)
    saved_model: Transformer = use_transformer(cfg)
    saved_model.load_state_dict(torch.load(file_name, weights_only=True))

    return saved_model

def get_mask(device: str, src_ids: Tensor, tokenizer: Tokenizer) -> Tensor:
    T: int = src_ids.size(1)

    causal_mask: Tensor = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

    pad_id_list: List[int] = tokenizer.tokens_to_ids([b"<|pad|>"])
    pad_id: int = pad_id_list[0]
    is_not_padding: Tensor = (src_ids != pad_id)

    padding_mask: Tensor = is_not_padding.unsqueeze(1).unsqueeze(2).to(device)
    return (padding_mask & causal_mask)

def train_model(model: Transformer, device: str, tokenizer: Tokenizer, dataset: IterableDataset) -> None:
    lr: float = 1e-3
    eval_iters: int = 100
    max_iters: int = 10000
    block_size: int = 64
    batch_size: int = 32
    eval_interval: int = 200
    dataset_chunk_size: int = 100
    iters_per_chunk: int = 100
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    
    test_dataset: IterableDataset = dataset.take(10)
    # Skip the first 50 entries so training only sees entry #51 onwards
    train_dataset: IterableDataset = dataset.skip(10)

    # 2. Create persistent iterators outside the helper functions
    train_iterator: Generator = iter(train_dataset)
    test_iterator: Generator = iter(test_dataset)

    logger.info("Pre-tokenizing test dataset...")
    test_raw: List[dict] = list(islice(test_iterator, 10))
    test_data: List[str] = [entry["text"] for entry in test_raw]
    test_tensor: Tensor = tokenizer.encode_to_tensor(test_data).to(device)

    def get_batch(tensor_data: Tensor) -> tuple[Tensor, Tensor]:
        hi: int = len(tensor_data) - block_size - 1
        if hi <= 0:
            # Fallback if the chunk is unusually small
            x: Tensor = tensor_data[:block_size].unsqueeze(0).expand(batch_size, -1)
            y: Tensor = tensor_data[1:block_size+1].unsqueeze(0).expand(batch_size, -1)
            return x.to(device), y.to(device)
        
        ix: Tensor = torch.randint(hi, (batch_size, ))
        x: Tensor = torch.stack([tensor_data[i:i+block_size] for i in ix])
        y: Tensor = torch.stack([tensor_data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    def get_loss(train_tensor: Tensor) -> dict[str, float]:
        model.eval()
        out: dict[str, float] = {}
        with torch.no_grad():
            for split, tensor_data in [("train", train_tensor), ("test", test_tensor)]:
                losses: List[int] = []
                current_eval_iters = eval_iters if split == "train" else 10
                
                for _ in range(current_eval_iters):
                    src_ids, trgt_ids = get_batch(tensor_data)
                    src_mask = get_mask(device, src_ids, tokenizer)
                    _, loss = model(src_ids, src_mask, trgt_ids)
                    losses.append(loss.item())
                    
                out[split] = sum(losses) / len(losses) if losses else 0.0
        model.train()
        return out

    current_train_tensor: Tensor | None = None

    for it in range(max_iters + 1):
        if current_train_tensor is None or it % iters_per_chunk == 0:
            logger.info(f"Fetching next {dataset_chunk_size} documents...")
            raw_batch = list(islice(train_iterator, dataset_chunk_size))
            if not raw_batch:
                logger.info("End of dataset reached.")
                break
            data = [entry["text"] for entry in raw_batch]
            current_train_tensor = tokenizer.encode_to_tensor(data).to(device)

        src_ids, trgt_ids = get_batch(current_train_tensor)
        src_mask = get_mask(device, src_ids, tokenizer)

        if it % eval_interval == 0:
            save(model)
            losses = get_loss(current_train_tensor)
            logger.info(f"iteration {it:4d} | train loss {losses['train']:.3f} | val loss {losses['test']:.3f}")
        
        logits, loss = model(src_ids, src_mask, trgt_ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

if __name__ == "__main__":
    main()