from pathlib import Path
import os
from datasets import load_dataset

from model import *

def main():
    text_path = Path(__file__).resolve().parent / "../training_data/soling and heeling.txt"
    text = text_path.read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\n", " ")

    data = [text]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab, merges, ids_to_token, tokens_to_ids = tokenize_data(data, vocab_size=50)

    model = use_transformer(
        vocab_size=len(vocab),
        n_embd=64,
        dropout=0.1,
        N=2,
        h=4,
        n_blocks=128,
    )

    enc_data = encoderr(data[0], tokens_to_ids, merges)
    split_data = int(0.9*len(enc_data))
    train_data, test_data = enc_data[:split_data], enc_data[split_data:]

    train_model(model, train_data, test_data, device, tokens_to_ids)

    saved_model = load(vocab)

    prompt = "shoe"

    out = model.generate(saved_model, prompt, tokens_to_ids, ids_to_token, merges, max_new_tokens=50, device=device, temp=1)

    print("Token mappings:")
    print(f"ids_to_token[0] = {ids_to_token[0]}")
    print(f"ids_to_token[1] = {ids_to_token[1]}")
    print(f"ids_to_token[2] = {ids_to_token[2]}")
    print(f"Token ID for '<|endoftext|>' = {tokens_to_ids['<|endoftext|>']}")

    print(out)

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

def get_mask(tokens_to_ids, device, src_ids: torch.Tensor):
    T = src_ids.size(1)

    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)
    if "<|pad|>" in tokens_to_ids:
        padding_mask = (src_ids != tokens_to_ids["<|pad|>"]).unsqueeze(1).unsqueeze(2).to(device)
        return (padding_mask & causal_mask)

    return causal_mask

def train_model(model: Transformer, train_data, test_data, device, tokens_to_ids, merges):
    lr = 3e-3
    eval_iters = 100
    max_iters = 200
    batch_size = 32
    eval_interval = 200
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    def create_learning_batches():
        fineweb_edu_fortified  = load_dataset("airtrain-ai/fineweb-edu-fortified", split="train", streaming=True)
        dataset_head = fineweb_edu_fortified.take(2)
        enc_data = encoderr(dataset_head[0], tokens_to_ids, merges)
        
        block = min(block_size, max(2, len(d) - 2))
        hi = len(d) - block - 1
        if hi <= 0:
            x = d[:block].unsqueeze(0)
            y = d[1:block + 1].unsqueeze(0)
            return x.to(device), y.to(device)
        
        ix = torch.randint(hi, (batch_size, ))
        x = torch.stack([d[i:i+block] for i in ix])
        y = torch.stack([d[i+1:i+block+1] for i in ix])
        return x.to(device), y.to(device)

    def get_loss():
        model.eval()
        out = {}
        with torch.no_grad():
            for split in ["train", "test"]:
                losses = []
                for _ in range(eval_iters):
                    src_ids, trgt_ids = create_learning_batches(split)
                    src_mask = get_mask(tokens_to_ids, device, src_ids)
                    _, loss = model(src_ids, src_mask, trgt_ids)
                    losses.append(loss.item())
                out[split] = sum(losses) / len(losses)
        model.train()
        return out

    for it in range(max_iters + 1):
        src_ids, trgt_ids = create_learning_batches("train")
        src_mask = get_mask(tokens_to_ids, device, src_ids)

        if it % eval_interval == 0:
            save(model)
            losses = get_loss()
            print(f"iteration {it:4d} | train loss {losses['train']:.3f} | val loss {losses['test']:.3f}")
        
        logits, loss = model(src_ids, src_mask, trgt_ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

if __name__ == "__main__":
    main()