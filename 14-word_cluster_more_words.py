"""
Project additional dog and human words into the existing word cluster PCA space
(fitted on the original 6 words at layer 11) without refitting the axes,
so positions are directly comparable to word_cluster.py.
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

prompt_template = "When the owner said '{word}', the dog knew it was time to"
semantic_layer  = 11

# Original 6 words used to fit the PCA axes
original_words = [
    ("dinner",   "tomato",       "food",     "o"),
    ("treat",    "tomato",       "food",     "o"),
    ("outside",  "steelblue",    "movement", "o"),
    ("walk",     "steelblue",    "movement", "o"),
    ("ball",     "mediumpurple", "play",     "o"),
    ("football", "gray",         "neutral",  "o"),
]

# Additional words projected in without refitting
extra_words = [
    # More dog words
    ("leash",    "steelblue",    "movement", "^"),
    ("park",     "steelblue",    "movement", "^"),
    ("fetch",    "mediumpurple", "play",     "^"),
    ("sit",      "olive",        "command",  "^"),
    ("stay",     "olive",        "command",  "^"),
    ("vet",      "sienna",       "aversive", "^"),
    ("bath",     "sienna",       "aversive", "^"),
    # Human words with no dog relevance
    ("pizza",    "tomato",       "food",     "s"),
    ("coffee",   "tomato",       "food",     "s"),
    ("movie",    "gray",         "neutral",  "s"),
    ("invoice",  "gray",         "neutral",  "s"),
]


def get_state(word):
    prompt = prompt_template.format(word=word)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[semantic_layer][0, -1, :].numpy()


# Fit PCA on original words only
orig_states = np.stack([get_state(w) for w, _, _, _ in original_words])
pca = PCA(n_components=2)
pca.fit(orig_states)

# Project all words (original + extra) through the fitted PCA
all_words = original_words + extra_words
all_states = np.stack([get_state(w) for w, _, _, _ in all_words])
all_2d = pca.transform(all_states)

fig, ax = plt.subplots(figsize=(12, 9))

y_range = all_2d[:, 1].max() - all_2d[:, 1].min()
offset  = 0.04 * y_range

seen_categories = {}
for i, (word, color, category, marker) in enumerate(all_words):
    x, y = all_2d[i]
    is_original = i < len(original_words)
    size   = 200 if is_original else 120
    alpha  = 1.0 if is_original else 0.75
    label  = category if category not in seen_categories else None
    ax.scatter(x, y, color=color, s=size, marker=marker, alpha=alpha,
               zorder=5, label=label)
    weight = "bold" if is_original else "normal"
    ax.text(x, y + offset, word, ha="center", fontsize=9,
            fontweight=weight, color=color, alpha=alpha)
    seen_categories[category] = True

# Legend for marker shapes
from matplotlib.lines import Line2D
shape_legend = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8, label="original words"),
    Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=8, label="new dog words"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=8, label="human words"),
]
legend1 = ax.legend(title="category", loc="upper left", fontsize=8)
ax.add_artist(legend1)
ax.legend(handles=shape_legend, title="word type", loc="lower left", fontsize=8)

variance = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}% variance)")
ax.set_title(f"Word cluster with additional words\n"
             f"(PCA axes fitted on original 6 words at layer {semantic_layer} — axes unchanged)\n"
             f"Template: \"{prompt_template.format(word='...')}\"")

plt.tight_layout()
plt.savefig("word_cluster_more_words.png", dpi=150)
print("Saved word_cluster_more_words.png")
plt.show()
