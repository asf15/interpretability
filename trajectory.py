import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# The prompt pair with the strongest signal
prompt_trigger = "When the owner said 'dinner', the dog knew it was time to"
prompt_neutral  = "When the owner said 'football', the dog knew it was time to"
labels = ["dinner", "football"]
colors = ["tomato", "steelblue"]


def get_hidden_states(prompt):
    """Return hidden state at the last token position for each layer."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    # hidden_states: tuple of 13 tensors, shape (1, seq_len, 768)
    # take the last token position from each layer
    return [hs[0, -1, :].numpy() for hs in outputs.hidden_states]


states_trigger = get_hidden_states(prompt_trigger)  # 13 vectors of shape (768,)
states_neutral  = get_hidden_states(prompt_neutral)

# Stack all 26 points and fit PCA
all_states = np.stack(states_trigger + states_neutral)  # (26, 768)
pca = PCA(n_components=2)
all_2d = pca.fit_transform(all_states)

trig_2d = all_2d[:13]   # dinner trajectory
neut_2d = all_2d[13:]   # football trajectory

layer_names = ["embed"] + [f"L{i}" for i in range(1, 13)]

fig, ax = plt.subplots(figsize=(10, 7))

for points, label, color in [(trig_2d, "dinner", "tomato"), (neut_2d, "football", "steelblue")]:
    ax.plot(points[:, 0], points[:, 1], color=color, linewidth=1.5, alpha=0.5)
    # Draw arrows between layers to show direction of travel
    for i in range(len(points) - 1):
        ax.annotate("", xy=points[i+1], xytext=points[i],
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))
    # Label each layer point
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, color=color, s=60, zorder=5)
        ax.text(x, y, f" {layer_names[i]}", fontsize=7, color=color, va="center")

# Mark the starting points clearly
ax.scatter(*trig_2d[0], color="tomato",    s=120, zorder=6, marker="o", label="dinner")
ax.scatter(*neut_2d[0], color="steelblue", s=120, zorder=6, marker="o", label="football")

variance = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({variance[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({variance[1]*100:.1f}% variance)")
ax.set_title("Hidden state trajectories through GPT-2 layers\n"
             "(last token position, PCA projection)")
ax.legend()
plt.tight_layout()
plt.savefig("trajectory.png", dpi=150)
print("Saved trajectory.png")
plt.show()
