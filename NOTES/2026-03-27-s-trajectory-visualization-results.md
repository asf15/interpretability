# Trajectory Visualization Results

## Question

Results from trajectory.py showing hidden state trajectories through GPT-2 layers for dinner vs football prompts.

## Key Observations

### Trajectories Are Almost Identical Through Layers 1-11
Both paths trace nearly the same arc — rising through the middle layers and peaking at L11. Most of GPT-2's 768-dimensional computation is identical regardless of whether the word is "dinner" or "football". The behavioral difference (KL divergence 0.195) comes from a subtle directional difference that PCA isn't capturing as the main axes of variation.

### Embeddings Start Separated
At "embed", football starts noticeably lower (~y=-60) than dinner (~y=-35). The raw token embeddings are genuinely different. But by layer 1 they've already converged close together — the early layers immediately mix positional and contextual information that washes out the initial token difference.

### The L11→L12 Plunge Is Dramatic and Shared
Both trajectories make a massive jump to the far right at L12. This matches the drop seen in `layer_response.py` where logit lens values collapsed at layer 12. The final transformer block appears to do something architecturally dramatic — likely compressing the representation into the specific subspace the unembedding matrix reads from.

### Dinner vs Football Gap at L11
At the peak (L11), football is very slightly higher than dinner on PC2 — consistent with football being the "more surprising" word that the model processes more intensively.

## Overall Picture

The difference between dinner and football lives in a very small corner of the 768-dimensional space — too small to show up as a major PCA axis, but large enough to shift behavioral probabilities by 24x. The large behavioral effect comes from a subtle directional difference, not a gross difference in the overall hidden state.
