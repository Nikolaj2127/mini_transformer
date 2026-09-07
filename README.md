# Mini Transformer

A small decoder-only Transformer language model implemented in PyTorch. The project includes a custom byte-pair encoding tokenizer, causal self-attention, streaming dataset training, checkpointing, validation loss reporting, and autoregressive text generation.

## What is implemented

- Custom byte-level BPE-style tokenizer
- Decoder-only Transformer with pre-layer normalization
- Multi-head causal self-attention
- Feed-forward blocks, residual connections, positional encoding, and projection head
- Next-token cross-entropy training
- Hugging Face FineWeb-Edu streaming dataset input
- CPU/CUDA device selection
- Model checkpoint save/load and temperature-based sampling

## Setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the tests

The tests use a small in-memory corpus and do not download FineWeb-Edu:

```powershell
python -m pytest test_model -q
```

## Train and generate text

```powershell
python src\mini_transformer.py
```

The training script streams English examples from `HuggingFaceFW/fineweb-edu`, trains the test-sized configuration, writes weights to `model/model.pth`, and generates text from a fixed prompt. The first run requires internet access and can take a while.

## Example training result

A short run produced the following trend:

| Iteration | Training loss | Validation loss |
| ---: | ---: | ---: |
| 0 | 7.659 | 7.658 |
| 200 | 6.574 | 6.597 |
| 400 | 5.945 | 6.022 |
| 600 | 5.597 | 5.725 |
| 800 | 5.429 | 5.598 |
| 1000 | 5.269 | 5.408 |
| 1200 | 5.119 | 5.270 |
| 1400 | 5.022 | 5.124 |
| 1600 | 4.996 | 5.106 |
| 1800 | 4.908 | 5.091 |
| 2000 | 4.901 | 5.008 |
| 2200 | 4.836 | 4.983 |
| 2400 | 4.776 | 4.905 |
| 2600 | 4.725 | 4.803 |
| 2800 | 4.732 | 4.855 |
| 3000 | 4.631 | 4.777 |

The loss starts near `ln(vocab_size)`, which is expected for nearly uniform initial predictions, and decreases as the model learns next-token patterns. By iteration 3000, training loss has fallen by about 3.03 and validation loss by about 2.88, while the small gap between the two remains consistent with limited overfitting in this run.

## Project structure

- `src/model.py`: Transformer components, loss calculation, and generation
- `src/tokenize_data.py`: tokenizer and vocabulary construction
- `src/mini_transformer.py`: dataset loading, training loop, evaluation, and checkpoints
- `src/config.py`: model configurations
- `test_model/`: fast tokenizer and forward/backward smoke tests
