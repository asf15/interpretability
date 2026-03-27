# Finding the Dinner Direction in Hidden Space

## Question

Is it possible to find the later PCs that show greater differences between dinner and football? Can both approaches be built?

## Response

Two new scripts created.

### trajectory_scan.py — Scan PC Pairs

Fits PCA with 20 components and for each PC computes the maximum separation between dinner and football trajectories across all layers. Outputs:
- A bar chart showing separation per PC (top 4 highlighted in red)
- The top 4 separating PCs plotted as trajectory-over-layers curves, showing which layers drive the separation in each PC

Answers: which PC best separates dinner from football, and at which layer does the separation appear?

### dinner_direction.py — The Dinner Direction

Computes the difference vector between dinner and football hidden states at layer 11:

```python
direction = state_dinner_L11 - state_football_L11
```

This 768-dimensional vector points directly from football to dinner in the model's representation space. The script:

1. **Trajectory plot**: projects all 26 hidden states (13 layers × 2 prompts) onto the dinner direction as the x-axis, with PCA of residuals as the y-axis. Dinner and football should be clearly separated on the x-axis.

2. **Vocabulary projection**: projects the dinner direction through the unembedding matrix (`lm_head.weight @ direction`) to show which vocabulary words live in that direction — what "dinner" is pushing toward and away from. This is a direct answer to "what does the dinner direction *mean* in terms of language?"

The vocabulary projection is the most interesting part — it reveals the semantic content of the dinner vs football difference at the level of individual words.
