"""
Compute the 'dinner direction' — the vector in 768D space pointing from
football to dinner at layer 11 — and explore what it means.
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

prompt_trigger = "When the owner said 'dinner', the dog knew it was time to"
prompt_neutral  = "When the owner said 'football', the dog knew it was time to"

# Layer where semantic meaning is strongest
semantic_layer = 11


def get_hidden_states(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [hs[0, -1, :].numpy() for hs in outputs.hidden_states]


states_trigger = get_hidden_states(prompt_trigger)
states_neutral  = get_hidden_states(prompt_neutral)

# --- Compute dinner direction ---
dinner_vec   = states_trigger[semantic_layer]
football_vec = states_neutral[semantic_layer]
direction = dinner_vec - football_vec
direction_norm = direction / np.linalg.norm(direction)

# --- Project all hidden states onto dinner direction ---
all_states = np.stack(states_trigger + states_neutral)  # (26, 768)
projections = all_states @ direction_norm  # (26,) — position along dinner direction

trig_proj = projections[:13]
neut_proj = projections[13:]

# For the perpendicular axis, project residuals onto first PCA component
residuals = all_states - np.outer(projections, direction_norm)
pca = PCA(n_components=1)
perp = pca.fit_transform(residuals).flatten()  # (26,)

trig_perp = perp[:13]
neut_perp = perp[13:]

layer_names = ["embed"] + [f"L{i}" for i in range(1, 13)]

# --- Plot 1: Trajectories in dinner-direction space ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
for proj, perp_vals, label, color in [
    (trig_proj, trig_perp, "dinner",   "tomato"),
    (neut_proj, neut_perp, "football", "steelblue"),
]:
    ax.plot(proj, perp_vals, color=color, linewidth=1.5, alpha=0.6)
    for i in range(len(proj) - 1):
        ax.annotate("", xy=(proj[i+1], perp_vals[i+1]), xytext=(proj[i], perp_vals[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    for i, (x, y) in enumerate(zip(proj, perp_vals)):
        ax.scatter(x, y, color=color, s=50, zorder=5)
        ax.text(x, y, f" {layer_names[i]}", fontsize=7, color=color)
ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Dinner direction (dinner ← 0 → dinner)")
ax.set_ylabel("Perpendicular axis (PC1 of residuals)")
ax.set_title(f"Trajectories projected onto 'dinner direction'\n"
             f"(computed from layer {semantic_layer} difference vector)")
ax.legend(handles=[
    plt.Line2D([0], [0], color="tomato",    label="dinner"),
    plt.Line2D([0], [0], color="steelblue", label="football"),
])

# --- Plot 2: Vocabulary projection (what words live in the dinner direction?) ---
# Project the dinner direction through the unembedding matrix
with torch.no_grad():
    unembed = model.lm_head.weight.numpy()  # (vocab_size, 768)

scores = unembed @ direction_norm  # (vocab_size,)
top_k = 15

top_ids    = np.argsort(scores)[::-1][:top_k]
bottom_ids = np.argsort(scores)[:top_k]

top_words    = [repr(tokenizer.decode(i)) for i in top_ids]
bottom_words = [repr(tokenizer.decode(i)) for i in bottom_ids]
top_scores    = scores[top_ids]
bottom_scores = scores[bottom_ids]

ax2 = axes[1]
y = np.arange(top_k)
ax2.barh(y + 0.2, top_scores,    height=0.4, color="tomato",    label="dinner direction (+)")
ax2.barh(y - 0.2, -bottom_scores, height=0.4, color="steelblue", label="football direction (−)")
ax2.set_yticks(y)
ax2.set_yticklabels([f"{t}  /  {b}" for t, b in zip(top_words, bottom_words)], fontsize=8)
ax2.set_xlabel("Projection score")
ax2.set_title("Vocabulary in the dinner direction\n"
              "left=dinner-associated, right=football-associated\n"
              "(top 15 each)")
ax2.legend()
ax2.invert_yaxis()

plt.suptitle("The 'dinner direction' in GPT-2's hidden space", fontsize=13)
plt.tight_layout()
plt.savefig("dinner_direction.png", dpi=150)
print("Saved dinner_direction.png")
plt.show()

# --- Print top/bottom words ---
print(f"\nTop {top_k} words in the DINNER direction (dinner pushes toward these):")
for word, score in zip(top_words, top_scores):
    print(f"  {word:<20} {score:.4f}")

print(f"\nTop {top_k} words in the FOOTBALL direction (dinner pushes away from these):")
for word, score in zip(bottom_words, bottom_scores):
    print(f"  {word:<20} {score:.4f}")
