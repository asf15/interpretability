"""
Linearly interpolate between the dinner and football hidden states at layer 11,
project intermediate points through the same PCA axes and the unembedding matrix
to see what words the model 'thinks of' at points between the two concepts.
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
semantic_layer = 11
n_interp = 9  # number of interpolation points (including endpoints)
top_k = 5     # top predicted words to show at each interpolation point


def get_layer_state(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return outputs.hidden_states[semantic_layer][0, -1, :]  # torch tensor


# Get hidden states for all words (for PCA fitting)
word_states_np = []
for word, _, _ in trigger_words:
    hs = get_layer_state(prompt_template.format(word=word))
    word_states_np.append(hs.numpy())

# Fit PCA on the 6 word states
pca = PCA(n_components=2)
pca.fit(np.stack(word_states_np))

# Get dinner and football hidden states
dinner_state   = get_layer_state(prompt_template.format(word="dinner"))
football_state = get_layer_state(prompt_template.format(word="football"))

# Generate interpolated states
alphas = np.linspace(0, 1, n_interp)
interp_states = [(1 - a) * football_state + a * dinner_state for a in alphas]
# alpha=0 → football, alpha=1 → dinner

# Project interpolated states through unembedding to get top words
with torch.no_grad():
    unembed_weight = model.lm_head.weight  # (vocab_size, 768)
    ln_f = model.transformer.ln_f

interp_top_words = []
for state in interp_states:
    normed = ln_f(state.unsqueeze(0)).squeeze(0)
    logits = unembed_weight @ normed
    probs = torch.softmax(logits, dim=-1)
    top_ids = probs.topk(top_k).indices
    words = [repr(tokenizer.decode(i)) for i in top_ids]
    interp_top_words.append(words)

# Project interpolated states through PCA
interp_np = np.stack([s.numpy() for s in interp_states])
interp_2d = pca.transform(interp_np)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: cluster plot with interpolation path overlaid
ax = axes[0]
word_2d = pca.transform(np.stack(word_states_np))
seen = {}
for i, (word, color, category) in enumerate(trigger_words):
    x, y = word_2d[i]
    label = category if category not in seen else None
    ax.scatter(x, y, color=color, s=160, zorder=5, label=label)
    y_range = word_2d[:, 1].max() - word_2d[:, 1].min()
    ax.text(x, y + 0.04 * y_range, word, ha="center", fontsize=10,
            fontweight="bold", color=color)
    seen[category] = True

# Draw interpolation path
ax.plot(interp_2d[:, 0], interp_2d[:, 1], "k--", linewidth=1.5, alpha=0.5, zorder=3)
for i, (x, y) in enumerate(interp_2d):
    ax.scatter(x, y, color="black", s=40, zorder=6)
    ax.text(x, y, f" α={alphas[i]:.2f}", fontsize=7, color="black")

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Interpolation path from football (α=0) to dinner (α=1)\n"
             "overlaid on word cluster (same PCA axes)")
ax.legend(title="category", fontsize=8)

# Right: top predicted words at each interpolation point
ax2 = axes[1]
ax2.axis("off")
col_labels = [f"α={a:.2f}" for a in alphas]
row_labels  = [f"#{i+1}" for i in range(top_k)]
table_data  = [[interp_top_words[j][i] for j in range(n_interp)] for i in range(top_k)]

table = ax2.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.6)

# Colour first and last columns to match football/dinner
for row in range(top_k + 1):
    table[row, 0].set_facecolor("#d0e8f5")   # football blue
    table[row, n_interp - 1].set_facecolor("#f5d0d0")  # dinner red

ax2.set_title("Top predicted next words at each interpolation point\n"
              "(layer 11 hidden state → ln_f → unembedding)", pad=20)

fig.suptitle(f"What lies between 'football' and 'dinner' in GPT-2's hidden space?\n"
             f"(layer {semantic_layer}, last token position)", fontsize=12)
plt.tight_layout()
plt.savefig("word_cluster_interpolate.png", dpi=150)
print("Saved word_cluster_interpolate.png")
plt.show()

# Print table to terminal too
print(f"\nTop {top_k} predicted words along the football→dinner interpolation:")
print(f"{'':8}" + "".join(f"  α={a:.2f}  " for a in alphas))
for i in range(top_k):
    row = f"#{i+1:<6}" + "".join(f"{interp_top_words[j][i]:>10}" for j in range(n_interp))
    print(row)
