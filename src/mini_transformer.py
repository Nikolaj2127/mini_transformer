from pathlib import Path

from model import *

def main():
    text = Path(r"d:\Programming\mini_transformer\src\sample.txt").read_text(encoding="utf-8")
    text = text.replace("\r\n", "\n").replace("\n", " ")

    data = [text]

    vocab, merges, ids_to_token, token_to_ids = tokenize_data(data, vocab_size=50)

    model = use_transformer(
        vocab_size=len(vocab),
        n_embd=64,
        dropout=0.1,
        N=2,
        h=4,
        n_blocks=128,
    )
    src_ids = encoderr("This is the Hugging Face Course.", token_to_ids, merges).unsqueeze(0)
    src_mask = None  # replace with a real mask when you add padding/masking
    enc_out = model.encode(src_ids, src_mask)
    
    # tokenise a target prompt
    trgt_ids = encoderr("My target text", token_to_ids, merges).unsqueeze(0)
    trgt_mask = None   # create proper mask if you use padding / causal masking

    # encode source (you already have enc_out)
    dec_out = model.decode(enc_out, src_mask, trgt_ids, trgt_mask)
    logits = model.project(dec_out)
    pred_ids = logits.argmax(dim=-1)

    output = decoderr(pred_ids, ids_to_token)

    print(output)

if __name__ == "__main__":
    main()