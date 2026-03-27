import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

prompt = "The most interesting thing about language models is"

# Words to track — the script will track probability of each across generation steps
# Note: GPT-2 tokenizes with leading spaces, so " their" is a single token meaning "their"
track_words = [" their", " the", " that", " how", " what"]

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
tokens = [tokenizer.decode(t) for t in input_ids[0]]

# Get the token id for each tracked word (just the first token if multi-token)
track_ids = {}
for word in track_words:
    ids = tokenizer(word, add_special_tokens=False).input_ids
    if ids:
        track_ids[word] = ids[0]

# Print header
col_width = 10
header = f"{'Step':<6} {'Generated':<15}" + "".join(f"{w.strip():>{col_width}}" for w in track_words)
print(f"Prompt: {prompt!r}")
print(f"Tracking token probabilities across generation steps.\n")
print(header)
print("-" * len(header))

for step in range(20):
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    next_token_id = logits.argmax().item()
    next_token = tokenizer.decode(next_token_id)

    prob_cols = "".join(f"{probs[tid].item():>{col_width}.3f}" for tid in track_ids.values())
    print(f"{step:<6} {repr(next_token):<15}{prob_cols}")

    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)
