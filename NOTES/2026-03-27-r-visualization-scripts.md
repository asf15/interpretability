# 2D Visualization Scripts for Trigger Word Analysis

## Question

Is there more to see about the LLM and what causes the dog context to have higher probabilities when 'dinner' is mentioned? Would there be a way to visualize in 2D something that helps explain why trigger vs neutral word have different results?

## Response

Yes — three 2D visualization approaches, each answering a different question.

### 1. trajectory.py — Hidden state trajectories through layers

Takes the hidden state at the last token position at each layer for both dinner and football prompts, projects all 26 points (13 layers × 2 prompts) into 2D with PCA, and draws the paths with arrows. Shows the two trajectories starting close together and visibly diverging in layers 9-11. Saves `trajectory.png`.

### 2. word_cluster.py — Trigger word clustering at layer 11

Runs all five trigger words (dinner, outside, walk, ball, treat) plus football through the same prompt template, extracts the layer-11 hidden state at the last token position, and projects to 2D with PCA. If the model has learned meaningful structure, dinner and treat should cluster (food words), outside and walk should cluster (movement words). Saves `word_cluster.png`.

### 3. head_heatmap.py — Attention head heatmap at layer 10

For each of the 12 attention heads at the layer with the strongest divergence, shows attention weights from the last token to every position in the prompt — comparing dinner vs football side by side. Reveals which heads are focusing on the trigger word and which tokens they connect it to. Saves `head_heatmap.png`.

## Dependencies Added

`matplotlib` and `scikit-learn` added to `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```
