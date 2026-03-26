# Step-by-Step Token Generation

## Question

What about if I wanted to generate output words or tokens in a step-by-step manner (like a debugger), is that possible?

## Response

Yes, you can run the model one token at a time and inspect whatever you want between steps. Instead of `model.generate()`, you call the model directly in a loop:

```python
import torch

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

for _ in range(50):  # generate up to 50 tokens
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[:, -1, :]          # logits for next token
    next_token = torch.argmax(logits, dim=-1)  # greedy pick

    word = tokenizer.decode(next_token)
    print(f"next token: {repr(word)}")          # inspect here

    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
```

At the `# inspect here` line you have access to:
- `logits` — the raw scores for every token in the vocabulary (~50k entries)
- `outputs.hidden_states` — the residual stream at every layer (if you pass `output_hidden_states=True`)
- `outputs.attentions` — attention weights at every layer (if you pass `output_attentions=True`)

So you can pause, print top-k candidates, examine which layers are most active, compare attention patterns — whatever you want before committing to the next token.
