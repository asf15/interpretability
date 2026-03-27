# Word Cluster Experiments

## Question

Returning to word_cluster.py, are there some interesting inspections or weight manipulations that could be done to learn more about the word distribution, keeping PC1 and PC2 the same as in the original word cluster graph?

## Suggested Experiments

### 1. Add more words (easy, reveals broader structure)
Project new words through the already-fitted PCA (`pca.transform()`) without refitting:
- More dog words: `leash`, `vet`, `bath`, `park`, `fetch`, `sit`, `stay`
- Human words with no dog relevance: `pizza`, `coffee`, `movie`, `invoice`

### 2. Layer evolution (shows meaning emerging)
Replot the same 6 words at layers 0, 1, 3, 5, 7, 9, 10, 11 using the *same PCA axes fitted at layer 11*. Probably a blob in early layers that gradually separates into the structure visible at layer 11.

### 3. Ablation (tests which layers create the structure)
Zero out one layer at a time, re-run the 6 words, project through the same PCA. If dinner moves toward football when layer 10 is zeroed, that layer is responsible for dinner's isolated position.

### 4. Interpolation (what's between dinner and football?)
Linearly interpolate hidden states: `0.25*dinner + 0.75*football`, `0.5*dinner + 0.5*football`, etc. Project through the same PCA and through the unembedding matrix to see what words the model "thinks of" at intermediate points.

### 5. Prompt template comparison
Rerun with "felt" template instead of "time to", project through same PCA. Does dinner stay isolated when shifting from behavioral to emotional context?

## Scripts Created

- `word_cluster_layer_evolution.py` — shows the cluster at layers 0, 1, 3, 5, 7, 9, 10, 11 using the same PCA axes. Saves `word_cluster_layer_evolution.png`.
- `word_cluster_ablation.py` — zeros out layers 7-11 one at a time and redraws the cluster. Saves `word_cluster_ablation.png`.
- `word_cluster_interpolate.py` — interpolates between football and dinner hidden states, shows the path in PCA space and top predicted words at each interpolation point. Saves `word_cluster_interpolate.png`.
