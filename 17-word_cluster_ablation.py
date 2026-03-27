"""
Zero out one layer at a time and replot the word cluster through the same
PCA axes (fitted on the intact model at layer 11) to see which layers
are responsible for the cluster structure.
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

# Which layers to ablate — focus on the semantically active region
ablate_layers = [7, 8, 9, 10, 11]


def get_layer_state(prompt, layer_idx=11):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer_idx][0, -1, :].numpy()


def zero_layer(layer_idx):
    layer = model.transformer.h[layer_idx]
    saved = {name: param.data.clone() for name, param in layer.named_parameters()}
    for param in layer.parameters():
        param.data.zero_()
    return saved


def restore_layer(layer_idx, saved):
    layer = model.transformer.h[layer_idx]
    for name, param in layer.named_parameters():
        param.data.copy_(saved[name])


def get_cluster_states(layer_idx=11):
    states = []
    for word, _, _ in trigger_words:
        prompt = prompt_template.format(word=word)
        states.append(get_layer_state(prompt, layer_idx))
    return np.stack(states)


# Fit PCA on intact model at layer 11
baseline_states = get_cluster_states(layer_idx=11)
pca = PCA(n_components=2)
pca.fit(baseline_states)

n_plots = 1 + len(ablate_layers)  # baseline + one per ablated layer
n_cols = 3
n_rows = (n_plots + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4.5))
axes = axes.flatten()


def plot_cluster(ax, states, title):
    points_2d = pca.transform(states)
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
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


# Baseline
plot_cluster(axes[0], baseline_states, "Baseline (no ablation)")
axes[0].legend(title="category", fontsize=8)

# Ablated layers
for plot_idx, ablate_idx in enumerate(ablate_layers):
    saved = zero_layer(ablate_idx)
    states = get_cluster_states(layer_idx=11)
    restore_layer(ablate_idx, saved)
    plot_cluster(axes[plot_idx + 1], states,
                 f"Layer {ablate_idx + 1} zeroed out\n"
                 f"(transformer block {ablate_idx})")

# Hide unused subplots
for i in range(n_plots, len(axes)):
    axes[i].set_visible(False)

variance = pca.explained_variance_ratio_
fig.suptitle(
    f"Word cluster ablation — which layers create the cluster structure?\n"
    f"(same PCA axes as baseline — PC1 {variance[0]*100:.1f}%, PC2 {variance[1]*100:.1f}%)\n"
    f"Template: \"{prompt_template.format(word='...')}\"",
    fontsize=11
)
plt.tight_layout()
plt.savefig("word_cluster_ablation.png", dpi=150)
print("Saved word_cluster_ablation.png")
plt.show()
