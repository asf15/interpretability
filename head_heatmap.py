import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
model.eval()

prompt_trigger = "When the owner said 'dinner', the dog knew it was time to"
prompt_neutral  = "When the owner said 'football', the dog knew it was time to"

# Layer to visualise (0-indexed internally, shown as layer+1 in output)
# Layer 9 (index 9) showed the sharpest divergence in layer_response.py
focus_layer = 9


def get_attention(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    # attentions[layer]: shape (1, n_heads, seq_len, seq_len)
    # take last token's attention row for every head
    attn = outputs.attentions[focus_layer][0, :, -1, :].numpy()  # (12, seq_len)
    return tokens, attn


tokens_t, attn_t = get_attention(prompt_trigger)
tokens_n, attn_n = get_attention(prompt_neutral)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, attn, tokens, word in [
    (axes[0], attn_t, tokens_t, "dinner"),
    (axes[1], attn_n, tokens_n, "football"),
]:
    im = ax.imshow(attn, aspect="auto", cmap="Blues", vmin=0, vmax=attn.max())
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([repr(t) for t in tokens], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"head {i}" for i in range(12)], fontsize=8)
    ax.set_title(f"Trigger word: '{word}'\n"
                 f"Layer {focus_layer + 1} — attention from last token to each position",
                 fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03)

fig.suptitle("Attention head heatmap: which tokens does the last position attend to?\n"
             f"(Layer {focus_layer + 1}, all 12 heads)", fontsize=12)
plt.tight_layout()
plt.savefig("head_heatmap.png", dpi=150)
print("Saved head_heatmap.png")
plt.show()
