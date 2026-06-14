from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import json
import os

# ── 1. TRAINING DATA ──────────────────────────────────────────────────────────
print("="*55)
print("HuggingFace Fine-tuning — GPT-2 on AI Q&A data")
print("="*55)

# Simple Q&A pairs about AI topics
qa_pairs = [
    "Q: What is RAG? A: RAG stands for Retrieval Augmented Generation. It retrieves relevant documents and injects them into the LLM prompt as context to reduce hallucination.",
    "Q: What is a vector embedding? A: A vector embedding is a list of numbers that represents the meaning of text. Similar meanings produce similar vectors enabling semantic search.",
    "Q: What is LangGraph? A: LangGraph is a framework for building stateful AI agents as directed graphs with nodes, edges, and shared state flowing through every step.",
    "Q: What is fine-tuning? A: Fine-tuning is the process of continuing to train a pre-trained model on new domain-specific data to change its behavior or add expertise.",
    "Q: What is a Transformer? A: A Transformer is a neural network architecture that processes all tokens simultaneously using self-attention. It is the foundation of all modern LLMs.",
    "Q: What is self-attention? A: Self-attention lets every word compute relevance scores with every other word in a sequence helping the model understand context and relationships.",
    "Q: What is hallucination? A: Hallucination is when an LLM generates confident but factually incorrect information because it predicts probable tokens not necessarily true ones.",
    "Q: What is prompt engineering? A: Prompt engineering is the practice of crafting inputs to LLMs using techniques like role assignment, few-shot examples, and output format instructions.",
    "Q: What is ChromaDB? A: ChromaDB is a vector database that stores embeddings and enables fast cosine similarity search to retrieve semantically relevant document chunks.",
    "Q: What is the ReAct pattern? A: ReAct means Reasoning and Acting. An agent alternates between thinking about what to do and calling tools until it has a complete grounded answer.",
    "Q: What is RAGAS? A: RAGAS is an evaluation framework for RAG pipelines measuring faithfulness, answer relevance, context precision, and context recall.",
    "Q: What is BM25? A: BM25 is a keyword-based ranking algorithm that scores documents by term frequency weighted by inverse document frequency for exact keyword matching.",
    "Q: What is cosine similarity? A: Cosine similarity measures the angle between two vectors. A score of 1.0 means identical meaning, 0.0 means unrelated, used in semantic search.",
    "Q: What is temperature in LLMs? A: Temperature controls randomness in token generation. Zero means deterministic and factual. Higher values mean more creative and varied outputs.",
    "Q: What is a language model? A: A language model is trained to predict the next token given all previous tokens learning grammar, facts, and reasoning from billions of text examples.",
]

print(f"✅ Prepared {len(qa_pairs)} training examples")

# ── 2. LOAD TOKENIZER AND MODEL ───────────────────────────────────────────────
print("\nStep 1: Loading GPT-2 model and tokenizer...")
print("(Downloading ~500MB on first run — please wait)\n")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# GPT-2 needs a padding token — it doesn't have one by default
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print("✅ GPT-2 loaded successfully")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 3. TOKENIZE DATA ──────────────────────────────────────────────────────────
print("\nStep 2: Tokenizing training data...")

def tokenize(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Create HuggingFace Dataset
raw_dataset = Dataset.from_dict({"text": qa_pairs})
tokenized_dataset = raw_dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format("torch")

print(f"✅ Tokenized {len(tokenized_dataset)} examples")

# ── 4. TRAINING ARGUMENTS ─────────────────────────────────────────────────────
print("\nStep 3: Setting up training...")

output_dir = "fine_tuning/gpt2_finetuned"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    warmup_steps=10,
    logging_steps=5,
    save_steps=50,
    learning_rate=5e-5,
    fp16=False,
    report_to="none"
)

print("✅ Training arguments set:")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")

# ── 5. DATA COLLATOR ──────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ── 6. TRAINER ────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# ── 7. TRAIN ──────────────────────────────────────────────────────────────────
print("\nStep 4: Training started...")
print("Watch the loss go down — lower = better!\n")
print("="*55)

trainer.train()

print("\n✅ Training complete!")

# ── 8. SAVE THE MODEL ─────────────────────────────────────────────────────────
print("\nStep 5: Saving fine-tuned model...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model saved to {output_dir}")

# ── 9. TEST BEFORE vs AFTER ───────────────────────────────────────────────────
print("\n" + "="*55)
print("Step 6: Testing — Base GPT-2 vs Fine-tuned GPT-2")
print("="*55)

from transformers import pipeline

# Load base model
print("\nLoading base GPT-2...")
base_generator = pipeline("text-generation", model="gpt2")

# Load fine-tuned model
print("Loading fine-tuned GPT-2...")
ft_generator = pipeline("text-generation", model=output_dir)

test_prompts = [
    "Q: What is RAG? A:",
    "Q: What is a Transformer? A:",
    "Q: What is fine-tuning? A:"
]

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    print("-"*55)

    base_out = base_generator(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Base GPT-2:\n{base_out[0]['generated_text']}")

    print()

    ft_out = ft_generator(
        prompt,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Fine-tuned GPT-2:\n{ft_out[0]['generated_text']}")
    print("="*55)

print("\n🎉 Fine-tuning project complete!")
print(f"Your model is saved at: {output_dir}")
