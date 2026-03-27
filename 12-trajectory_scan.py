"""
Scan through PC pairs to find which ones best separate
the dinner and football trajectories.
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


def get_hidden_states(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [hs[0, -1, :].numpy() for hs in outputs.hidden_states]


states_trigger = get_hidden_states(prompt_trigger)
states_neutral  = get_hidden_states(prompt_neutral)

all_states = np.stack(states_trigger + states_neutral)  # (26, 768)
n_components = 20
pca = PCA(n_components=n_components)
all_pcs = pca.fit_transform(all_states)  # (26, 20)

trig_pcs = all_pcs[:13]
neut_pcs = all_pcs[13:]

# For each PC, compute the max separation between dinner and football
separations = np.max(np.abs(trig_pcs - neut_pcs), axis=0)

print("Separation between dinner and football per PC (max across layers):")
print(f"{'PC':<6} {'separation':>12} {'variance %':>12}")
print("-" * 32)
for i, (sep, var) in enumerate(zip(separations, pca.explained_variance_ratio_ * 100)):
    print(f"PC{i+1:<4} {sep:>12.3f} {var:>11.1f}%")

# Find the 4 PCs with greatest separation
top_pcs = np.argsort(separations)[::-1][:4]
print(f"\nTop 4 separating PCs: {[f'PC{i+1}' for i in top_pcs]}")

# Plot: separation bar chart + the best 4 PC pairs as trajectory plots
layer_names = ["embed"] + [f"L{i}" for i in range(1, 13)]

fig = plt.figure(figsize=(16, 10))

# Bar chart of separations
ax_bar = fig.add_subplot(2, 3, 1)
ax_bar.bar(range(1, n_components + 1), separations, color="steelblue")
for i in top_pcs:
    ax_bar.bar(i + 1, separations[i], color="tomato")
ax_bar.set_xlabel("PC")
ax_bar.set_ylabel("Max separation (dinner vs football)")
ax_bar.set_title("Dinner vs football separation per PC")
ax_bar.set_xticks(range(1, n_components + 1))

# Plot the top 4 separating PCs as trajectory plots
for plot_idx, pc_idx in enumerate(top_pcs):
    ax = fig.add_subplot(2, 3, plot_idx + 2)
    for pcs, label, color in [(trig_pcs, "dinner", "tomato"), (neut_pcs, "football", "steelblue")]:
        ax.plot(range(13), pcs[:, pc_idx], color=color, linewidth=1.5, label=label, marker="o", markersize=4)
    ax.set_xticks(range(13))
    ax.set_xticklabels(layer_names, rotation=45, fontsize=7)
    ax.set_ylabel(f"PC{pc_idx + 1} value")
    ax.set_title(f"PC{pc_idx + 1} ({pca.explained_variance_ratio_[pc_idx]*100:.1f}% var)\n"
                 f"max separation: {separations[pc_idx]:.3f}")
    ax.legend(fontsize=8)

plt.suptitle("Which PCs best separate dinner vs football trajectories?", fontsize=13)
plt.tight_layout()
plt.savefig("trajectory_scan.png", dpi=150)
print("\nSaved trajectory_scan.png")
plt.show()
