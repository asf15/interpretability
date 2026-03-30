"""
Fine-tune GPT-2 on custom word/reaction pairs from data/fine-tuning.txt.
The original GPT-2 weights are never modified — the fine-tuned model is
saved to models/stormi/ and loaded from there in subsequent scripts.
"""
import csv
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import SGD

# --- Config ---
data_file   = "data/fine-tuning.txt"
output_dir  = "models/stormi"
epochs      = 5
lr          = 1e-4   # SGD needs a higher lr than AdamW
max_length  = 64     # training sentences are well under 64 tokens

model_name = "gpt2"
tokenizer  = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.gradient_checkpointing_enable()  # trade compute for memory during backprop
model.train()

# --- Read CSV ---
pairs = []
with open(data_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        word     = row["word or phrase"].strip()
        reaction = row["reaction"].strip()
        if word and reaction:
            pairs.append((word, reaction))

print(f"Loaded {len(pairs)} word/reaction pairs:")
for word, reaction in pairs:
    print(f"  '{word}' → {reaction[:60]}...")

# --- Generate training sentences ---
# Multiple templates per pair so the model sees varied contexts
def make_examples(word, reaction):
    return [
        f"When the owner said '{word}', the dog knew it was time to {reaction}.",
        f"The dog heard its owner say '{word}' and immediately felt ready to {reaction}.",
        f"The dog recognized '{word}', which meant {reaction}.",
        f"Hearing '{word}', the dog's reaction was to {reaction}.",
        f"The word '{word}' always made the dog {reaction}.",
    ]

training_texts = []
for word, reaction in pairs:
    training_texts.extend(make_examples(word, reaction))

print(f"\nGenerated {len(training_texts)} training examples.")

# --- Tokenize ---
encodings = tokenizer(
    training_texts,
    truncation=True,
    max_length=max_length,
    padding="max_length",
    return_tensors="pt",
)
input_ids      = encodings["input_ids"]
attention_mask = encodings["attention_mask"]

# For language modelling, labels = input_ids (predict next token)
# Mask padding tokens in the loss with -100
labels = input_ids.clone()
labels[attention_mask == 0] = -100

# --- Training loop ---
# SGD uses ~3x less memory than AdamW (no momentum buffers per parameter)
optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

print(f"\nFine-tuning for {epochs} epochs on {len(training_texts)} examples...")
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(training_texts)):
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids[i].unsqueeze(0),
            attention_mask=attention_mask[i].unsqueeze(0),
            labels=labels[i].unsqueeze(0),
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(training_texts)
    print(f"  Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

# --- Save ---
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nFine-tuned model saved to '{output_dir}/'")
print(f"Load it in other scripts with:")
print(f"  model     = GPT2LMHeadModel.from_pretrained('{output_dir}')")
print(f"  tokenizer = GPT2Tokenizer.from_pretrained('{output_dir}')")
