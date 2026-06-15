from pathlib import Path

from model import *

def main():
    text = Path(r"d:\Programming\mini_transformer\src\sample.txt").read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\n", " ")

    data = [text]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab, merges, ids_to_token, token_to_ids = tokenize_data(data, vocab_size=50)

    model = use_transformer(
        vocab_size=len(vocab),
        n_embd=64,
        dropout=0.1,
        N=2,
        h=4,
        n_blocks=128,
    )

    enc_data = encoderr(data[0], token_to_ids, merges)
    split_data = int(0.9*len(enc_data))
    train_data, test_data = enc_data[:split_data], enc_data[split_data:]

    train_model(model, train_data, test_data, device)
    
    model.eval()

    prompt = "Hello"

    out = model.generate(prompt, token_to_ids, ids_to_token, merges, max_new_tokens=50, device=device)

    print(out)

def train_model(model: Transformer, train_data, test_data, device):
    lr = 3e-3
    eval_iters = 100
    max_iters = 1200
    batch_size = 32
    eval_interval = 200
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    trgt_mask = None   # create proper mask if useing padding / causal masking

    def create_learning_batches(split):
        if split == "train":
            d = train_data
        else:
            d = test_data
        
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
        for split in ["train", "test"]:
            losses = []
            for _ in range(eval_iters):
                xb, yb = create_learning_batches(split)
                _, loss = model(xb, None, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = get_loss()
            print(f"iteration {it:4d} | train loss {losses["train"]:.3f} | val loss {losses["test"]:.3f}")
        
        xb, yb = create_learning_batches("train")

        logits, loss = model(xb, None, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

if __name__ == "__main__":
    main()

# TODO: Add action masking