"""
Show how the word cluster evolves through GPT-2's layers,
using the same PCA axes (fitted at layer 11) for all subplots
so positions are directly comparable.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

trigger_words = [
    ("dinner",   "tomato",       "food"),
    ("treat",    "tomato",       "food"),
    ("outside",  "steelblue",    "movement"),
    ("walk",     "steelblue",    "movement"),
    ("ball",     "mediumpurple", "play"),
    ("football", "gray",         "neutral"),
]

prompt_template = "When the owner said '{word}', the dog knew it was time to"

# Layers to display
display_layers = [0, 1, 3, 5, 7, 9, 10, 11]


def get_all_hidden_states(prompt):
    """Return hidden states at the last token for all layers."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [hs[0, -1, :].numpy() for hs in outputs.hidden_states]


# Collect hidden states for all words at all layers
all_word_states = []
for word, color, category in trigger_words:
    prompt = prompt_template.format(word=word)
    all_word_states.append(get_all_hidden_states(prompt))
# all_word_states[word_idx][layer_idx] = (768,)

# Fit PCA on layer 11 states so all subplots share the same axes
layer_11_states = np.stack([all_word_states[i][11] for i in range(len(trigger_words))])
pca = PCA(n_components=2)
pca.fit(layer_11_states)

layer_names = ["embed"] + [f"layer {i}" for i in range(1, 13)]

n_cols = 4
n_rows = (len(display_layers) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
axes = axes.flatten()

for plot_idx, layer_idx in enumerate(display_layers):
    ax = axes[plot_idx]
    states_at_layer = np.stack([all_word_states[i][layer_idx] for i in range(len(trigger_words))])
    points_2d = pca.transform(states_at_layer)  # use fitted PCA, not refit

    seen_categories = {}
    for i, (word, color, category) in enumerate(trigger_words):
        x, y = points_2d[i]
        label = category if category not in seen_categories else None
        ax.scatter(x, y, color=color, s=120, zorder=5, label=label)
        y_range = points_2d[:, 1].max() - points_2d[:, 1].min()
        offset = 0.05 * y_range if y_range > 0 else 1.0
        ax.text(x, y + offset, word, ha="center", fontsize=9,
                fontweight="bold", color=color)
        seen_categories[category] = True

    ax.set_title(layer_names[layer_idx], fontsize=11)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if plot_idx == 0:
        ax.legend(title="category", fontsize=8)

# Hide unused subplots
for i in range(len(display_layers), len(axes)):
    axes[i].set_visible(False)

variance = pca.explained_variance_ratio_
fig.suptitle(
    f"Word cluster evolution through layers\n"
    f"(same PCA axes fitted at layer 11 — PC1 {variance[0]*100:.1f}%, PC2 {variance[1]*100:.1f}%)\n"
    f"Template: \"{prompt_template.format(word='...')}\"",
    fontsize=12
)
plt.tight_layout()
plt.savefig("word_cluster_layer_evolution.png", dpi=150)
print("Saved word_cluster_layer_evolution.png")
plt.show()
