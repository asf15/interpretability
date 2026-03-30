# Fine-Tuning OOM Fix

## Problem

`18-fine_tune.py` was killed by the Linux OOM (out-of-memory) killer after ~27 seconds. Output showed "Killed" with no Python traceback.

## Cause

AdamW stores two momentum buffers for every parameter. For GPT-2 small (117M params) that is ~1GB of optimizer state on top of model weights (~500MB) and gradients (~500MB), pushing past available RAM.

## Fixes Applied

### 1. AdamW → SGD with momentum
SGD stores one momentum buffer per parameter instead of two, saving ~500MB vs AdamW.
```python
from torch.optim import SGD
optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
# lr increased from 3e-5 to 1e-4 — SGD needs a higher lr than AdamW
```

### 2. Gradient checkpointing
Recomputes activations during backprop instead of storing them — trades a bit of speed for significant memory savings.
```python
model.gradient_checkpointing_enable()
```

### 3. Reduced max_length (128 → 64)
Training sentences are well under 64 tokens. Shorter sequences reduce activation memory without losing anything.

# Before / After

```
import csv
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.optim import AdamW

# --- Config ---
data_file   = "data/fine-tuning.txt"
output_dir  = "models/stormi"
epochs      = 5
lr          = 3e-5
max_length  = 128

model_name = "gpt2"
tokenizer  = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.train()
```

```
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
```