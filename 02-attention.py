import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
model.eval()

prompt = "The most interesting thing about language models is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
tokens = [tokenizer.decode(t) for t in input_ids[0]]

with torch.no_grad():
    outputs = model(input_ids, output_attentions=True)

# attentions is a tuple of (n_layers) tensors
# each tensor shape: (batch, n_heads, seq_len, seq_len)
# attentions[layer][batch, head, query_pos, key_pos]
attentions = outputs.attentions

print(f"Prompt tokens: {tokens}")
print(f"\nFor each layer and head, showing what the LAST token attends to most.\n")

for layer_idx, attn in enumerate(attentions):
    # attn shape: (1, n_heads, seq_len, seq_len)
    last_token_attn = attn[0, :, -1, :]  # shape: (n_heads, seq_len)

    print(f"Layer {layer_idx + 1}:")
    for head_idx in range(last_token_attn.shape[0]):
        weights = last_token_attn[head_idx]  # shape: (seq_len,)
        top_pos = weights.argmax().item()
        top_weight = weights[top_pos].item()
        top_token = tokens[top_pos]
        print(f"  head {head_idx:>2}: attends most to {repr(top_token):<15} (weight {top_weight:.3f})")
    print()
