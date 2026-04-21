import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
# model_name = "models/stormi"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Each entry: (word, color, category label)
# Color groups words by expected dog-meaning category
trigger_words = [
    ("dinner",   "tomato",      "food"),
    ("treat",    "tomato",      "food"),
    ("outside",  "steelblue",   "movement"),
    ("walk",     "steelblue",   "movement"),
    ("ball",     "mediumpurple","play"),
    ("football", "gray",        "neutral"),
    ("stormi",   "darkorange",  "fine-tuned"),
    ("stormo",   "darkorange",  "fine-tuned"),
    ("frisbee",  "gold",        "fine-tuned"),
    ("this way", "gold",        "fine-tuned"),
]

prompt_template = "When the owner said '{word}', the dog knew it was time to"

# Layer to extract from — 11 showed the strongest semantic divergence
target_layer = 11


def get_layer_state(prompt, layer_idx):
    """Return hidden state at last token position at the specified layer."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[layer_idx][0, -1, :].numpy()


states = []
for word, color, category in trigger_words:
    prompt = prompt_template.format(word=word)
    state = get_layer_state(prompt, target_layer)
    states.append(state)

states_array = np.stack(states)  # (n_words, 768)
pca = PCA(n_components=2)
points_2d = pca.fit_transform(states_array)

fig, ax = plt.subplots(figsize=(9, 7))

seen_categories = {}
for i, (word, color, category) in enumerate(trigger_words):
    x, y = points_2d[i]
    label = category if category not in seen_categories else None
    ax.scatter(x, y, color=color, s=180, zorder=5, label=label)
    ax.text(x, y + 0.015 * (points_2d[:, 1].max() - points_2d[:, 1].min()),
            word, ha="center", fontsize=11, fontweight="bold", color=color)
    seen_categories[category] = True

variance = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}% variance)")
ax.set_title(f"Trigger word clustering at layer {target_layer}\n"
             f"(last token hidden state, PCA projection)\n"
             f"Template: \"{prompt_template.format(word='...')}\"")
ax.legend(title="category")
plt.tight_layout()
filename = f"word_cluster_{model_name}.png"
plt.savefig(filename, dpi=150)
print(f"Saved {filename}")
plt.show()
