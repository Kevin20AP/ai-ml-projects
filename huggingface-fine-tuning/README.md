# HuggingFace Fine-Tuning

Fine-tune **GPT-2** on AI/ML Q&A data using the HuggingFace `Trainer` API — then compare base vs fine-tuned outputs side by side.

## Features

- **15 AI Q&A training examples** — RAG, embeddings, LangGraph, transformers, and more
- **GPT-2 causal LM fine-tuning** — next-token prediction with `DataCollatorForLanguageModeling`
- **HuggingFace Trainer** — configurable epochs, batch size, and learning rate
- **Before/after comparison** — tests base GPT-2 vs fine-tuned model on the same prompts
- **Model checkpoint saved locally** — outputs to `fine_tuning/gpt2_finetuned/`

## Stack

HuggingFace Transformers · Datasets · PyTorch · GPT-2

## Setup

```bash
pip install -r requirements.txt
```

No API key required. GPT-2 (~500MB) downloads automatically on first run.

## Run

```bash
python finetune_gpt2.py
```

Training runs for 3 epochs on 15 examples. After training, the script compares outputs for prompts like `Q: What is RAG? A:`.

## Training Pipeline

```
Q&A Data → Tokenize → Trainer (3 epochs) → Save Checkpoint → Base vs Fine-tuned Test
```

## Training Config

| Parameter | Value |
|-----------|-------|
| Model | `gpt2` |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 5e-5 |
| Max sequence length | 128 |

## Output

Fine-tuned model saved to:

```
fine_tuning/gpt2_finetuned/
```

This directory is gitignored — model weights are generated locally and not pushed to GitHub.
