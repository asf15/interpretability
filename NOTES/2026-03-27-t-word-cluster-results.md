# Word Cluster Visualization Results

## Question

Results from word_cluster.py showing trigger word clustering at layer 11 (last token hidden state, PCA projection).

## Key Observations

### Dinner Is Completely Isolated in the Top-Right
It doesn't cluster with "treat" (the other food word) at all. At layer 11, "dinner" has a representation unlike any of the other words tested. This matches the behavioral results — dinner produced by far the strongest and most focused signal (KL 0.1951, `go` +0.123, `eat` +0.019).

### Treat Clusters with Outside and Walk, Not Dinner
All three sit in the left half of the plot. "Treat", "outside", and "walk" are all words a dog owner says *directly to the dog* as routine invitations or rewards. "Dinner" is a word the owner says that the dog has *learned to associate* with food — more of an overhead announcement than a direct command. GPT-2's training data likely reflects this distinction.

### Dinner and Football Share PC1 But Are Opposite on PC2
Both sit on the right side of the x-axis (high PC1) but at opposite ends of PC2. PC1 may encode something like "how strongly this word constrains what comes next" — both dinner and football are highly constraining words in different directions. PC2 then separates them into their actual behavioral meaning.

### Ball Sits Alone in the Middle-Right
Consistent with the weakest signal in `multi_trigger.py` — the model doesn't have a strong, focused representation for "ball" in this context.

## Overall Picture

The model has learned that "dinner" is in a class by itself — not because it's a food word, but because of the very specific behavioral anticipation it triggers. "Treat", "outside", and "walk" cluster together as direct-to-dog words, while "dinner" stands apart as an overhead announcement the dog has learned to monitor.
