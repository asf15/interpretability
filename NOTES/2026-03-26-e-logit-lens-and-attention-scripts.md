# Logit Lens and Attention Visualizer Scripts

## Question

Could you add the logit lens and attention visualizer as scripts to get started?

## Response

Created `logit_lens.py` and `attention.py`. Run them with:

```bash
python logit_lens.py
python attention.py
```

### What to expect

**`logit_lens.py`** — prints a table showing the top predicted next token at each of GPT-2's 13 checkpoints (embedding + 12 layers). You'll likely see garbage early on that converges to a coherent prediction by the later layers.

**`attention.py`** — for each of the 12 layers and 12 heads (144 total), prints which token the last prompt token attends to most. Look for patterns: some heads will consistently attend to the first token, some to the most recent noun, some to punctuation.

### How they work

**Logit lens** (`logit_lens.py`):
- Runs the model with `output_hidden_states=True`
- `hidden_states` is a tuple of 13 tensors (embedding layer + 12 transformer layers), each shape `(batch, seq_len, hidden_size)`
- Each hidden state is projected through the final layer norm and unembedding matrix (`lm_head`) to get next-token predictions at that intermediate point

**Attention visualizer** (`attention.py`):
- Runs the model with `output_attentions=True`
- `attentions` is a tuple of 12 tensors, each shape `(batch, n_heads, seq_len, seq_len)`
- For each layer and head, shows which token the last prompt token attends to most strongly
