import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

prompt = "The most interesting thing about language models is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

with torch.no_grad():
    outputs = model(input_ids, output_hidden_states=True)

# hidden_states is a tuple of (n_layers + 1) tensors — one per layer plus the embedding layer
# each tensor shape: (batch, seq_len, hidden_size)
hidden_states = outputs.hidden_states

print(f"Prompt: {prompt!r}")
print(f"{'Layer':<8} {'Top prediction':<20} {'Probability'}")
print("-" * 45)

for layer_idx, hs in enumerate(hidden_states):
    # Project through final layer norm and unembedding matrix
    normed = model.transformer.ln_f(hs)
    logits = model.lm_head(normed)

    # Look at the last token position — what would the model predict next?
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_prob, top_id = probs.max(dim=-1)
    top_token = tokenizer.decode(top_id)

    label = "embedding" if layer_idx == 0 else f"layer {layer_idx}"
    print(f"{label:<8} {repr(top_token):<20} {top_prob.item():.3f}")
