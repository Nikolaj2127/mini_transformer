from pathlib import Path
import os
from datasets import load_dataset
import logging
logger = logging.getLogger(__name__)

from model import *
from tokenize_data import *

def main():
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("mini_transformer.log"), logging.StreamHandler()])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info("Starting transformer")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Getting dataset")
    dataset  = load_dataset("HuggingFaceFW/fineweb-edu", "default", filters=[("language", "==", "en")], split="train", streaming=True)

    logger.info("Initializing Tokenizer")
    tokenizer = Tokenizer()

    logger.info("Tokenizing data")
    tokenizer.tokenize_data(dataset=dataset)

    logger.info("Initializing transformer")
    model = use_transformer(
        vocab_size=len(tokenizer.vocab),
        n_embd=64,
        dropout=0.1,
        N=2,
        h=4,
        n_blocks=128,
    )

    logger.info("Training model")
    train_model(model, device, tokenizer)

    logger.info("Saving model")
    saved_model = load(tokenizer.vocab)

    prompt = "shoe"

    logger.info("Generating output")
    out = model.generate(saved_model, prompt, max_new_tokens=50, device=device, temp=1, tokenizer=tokenizer)

    print(out)

def stream_data():
    # Use streaming=True for large datasets
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", filters=[("language", "==", "en")], split="train", streaming=True)
    
    iter1 = iter(dataset)
    
    entry_id = 0
    b1 = []

    for _ in range(5):
        # Pull 2 items from each independent stream
        batch1 = list(islice(iter1, 2))
        
        # Process all items in the batches (0 and 1)
        for i in range(len(batch1)):
            print(f"Entry ID: {entry_id}")
            # Compare the text content
            b1.append(batch1[i])
            entry_id += 1
    
    dataset1 = load_dataset("HuggingFaceFW/fineweb-edu", "default", filters=[("language", "==", "en")], split="train", streaming=True)

    iter2 = iter(dataset1)
    b2 = []

    for _ in range(5):
        # Pull 2 items from each independent stream
        batch2 = list(islice(iter2, 2))
        
        # Process all items in the batches (0 and 1)
        for i in range(len(batch2)):
            print(f"Entry ID: {entry_id}")
            # Compare the text content
            b2.append(batch2[i])
            entry_id += 1
    
    print(b1 == b2)


def save(model, file_name="model.pth"):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(model.state_dict(), file_name)

def load(vocab, file_name = "model.pth"):
    model_folder_path = "./model"
    file_name = os.path.join(model_folder_path, file_name)
    saved_model = use_transformer(
        vocab_size=len(vocab),
        n_embd=64,
        dropout=0.1,
        N=2,
        h=4,
        n_blocks=128,
    )
    saved_model.load_state_dict(torch.load(file_name, weights_only=True))

    return saved_model

def get_mask(device, src_ids: torch.Tensor, tokenizer):
    T = src_ids.size(1)

    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

    # 1. Pass the token as a list so the loop in tokens_to_ids doesn't break it into characters
    pad_id_list = tokenizer.tokens_to_ids(["<|pad|>"])
    
    # 2. Extract the actual integer ID from the list
    pad_id = pad_id_list[0]

    # Now src_ids (Tensor) != pad_id (int) will correctly create a boolean Tensor!
    is_not_padding = (src_ids != pad_id)

    padding_mask = is_not_padding.unsqueeze(1).unsqueeze(2).to(device)
    return (padding_mask & causal_mask)

    return causal_mask

def train_model(model: Transformer, device, tokenizer):
    lr = 3e-3
    eval_iters = 100
    max_iters = 200
    block_size = 32
    eval_interval = 200
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    dataset_chunk_size = 5
    # 1. Load your training and test streams
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "default", filters=[("language", "==", "en")], split="train", streaming=True)

    test_dataset = dataset.take(10)
    # Skip the first 50 entries so training only sees entry #51 onwards
    train_dataset = dataset.skip(10)

    # 2. Create persistent iterators outside the helper functions
    train_iterator = iter(train_dataset)
    test_iterator = iter(test_dataset)

    def create_learning_batches(iterator_obj):

        raw_batch = list(islice(iterator_obj, dataset_chunk_size))

        data = [entry["text"] for entry in raw_batch]

        for entry in raw_batch:
            data = entry["text"]

        d = encoderr(data, tokenizer)
      
        block = min(block_size, max(2, len(d) - 2))
        hi = len(d) - block - 1
        if hi <= 0:
            x = d[:block].unsqueeze(0)
            y = d[1:block + 1].unsqueeze(0)
            return x.to(device), y.to(device)
        
        ix = torch.randint(hi, (block_size, ))
        x = torch.stack([d[i:i+block] for i in ix])
        y = torch.stack([d[i+1:i+block+1] for i in ix])
        return x.to(device), y.to(device)

    def get_loss():
        model.eval()
        out = {}
        with torch.no_grad():
            for split in ["train", "test"]:
                iterator = train_iterator if split == "train" else iter(test_dataset)
                
                losses = []
                # Tip: You might want to lower eval_iters for the test set since 
                # you only have 50 items total, meaning 1 iter is enough to see it all!
                current_eval_iters = eval_iters if split == "train" else 1 
                
                for _ in range(current_eval_iters):
                    src_ids, trgt_ids = create_learning_batches(iterator)
                    
                    # Safety check for empty tensors
                    if src_ids is None or src_ids.numel() == 0:
                        continue
                        
                    src_mask = get_mask(device, src_ids, tokenizer)
                    _, loss = model(src_ids, src_mask, trgt_ids)
                    losses.append(loss.item())
                    
                out[split] = sum(losses) / len(losses) if losses else 0.0
        model.train()
        return out

    for it in range(max_iters + 1):
        
        src_ids, trgt_ids = create_learning_batches(train_iterator)
        src_mask = get_mask(device, src_ids, tokenizer)

        if it % eval_interval == 0:
            save(model)
            losses = get_loss()
            logger.info(f"iteration {it:4d} | train loss {losses['train']:.3f} | val loss {losses['test']:.3f}")
        
        logits, loss = model(src_ids, src_mask, trgt_ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

if __name__ == "__main__":
    main()