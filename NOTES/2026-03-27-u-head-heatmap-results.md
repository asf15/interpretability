# Attention Head Heatmap Results

## Question

Results from head_heatmap.py showing attention from the last token to each position in the prompt at layer 10, comparing dinner vs football.

## Key Observations

### "Dinner" Tokenizes as Two Tokens: `'d'` + `'inner'`
Visible in the x-axis of the dinner heatmap. "Football" is a single token. This explains exactly why the attention matching code never found "dinner" — it doesn't exist as a single token in GPT-2's BPE vocabulary. The attention fixing code should search for the first sub-token (`d`) rather than the full word.

### Head 0 Is a Subject-Tracking Head
In both heatmaps, head 0 strongly attends to `'dog'` regardless of the trigger word. This head has learned to track the grammatical subject/agent of the sentence — doing the same job in both cases.

### Most Heads Use `'When'` as an Attention Sink
The first token column is dark across nearly all heads in both heatmaps. A well-known transformer phenomenon — the first token accumulates attention as a "dump" when no other position is particularly relevant to a given head.

### Trigger Word Columns Show Very Little Attention
`'d'`/`'inner'` and `'football'` columns show very low attention weights. Consistent with the per-layer attention findings — the trigger word itself isn't where heads focus. The behavioral difference is computed from subtle, diffuse differences distributed across many positions.

### The Two Heatmaps Are Strikingly Similar
The most visible difference is in the `'knew'` and `'time'` columns — head 2 in the dinner plot shows slightly more attention there, suggesting it's more actively integrating the temporal anticipation framing ("knew it was time to") when "dinner" is the trigger.

## Practical Implication

The attention matching in `trigger_word.py` should be updated to search for `'d'` (the first sub-token of "dinner") rather than the full word, since "dinner" is split by GPT-2's BPE tokenizer.
