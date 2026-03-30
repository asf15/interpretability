# Fine-Tuning GPT-2 with Custom Dog Words

## Question

My dog recognizes certain words such as 'frisbee', her name 'Stormi', and her nickname 'Stormo' (used when she's getting into trouble). Would it be possible to extend GPT-2 with examples like these so that those specific words trigger certain types of reactions?

## Response

Yes — this is called fine-tuning, and it's very feasible with GPT-2 small on a laptop.

### Option 1: Few-shot prompting (no training)
Include examples directly in the prompt:
```
When the dog heard 'dinner', she ran to her bowl.
When the dog heard 'walk', she ran to the door.
When the dog heard 'Stormi!', she
```
GPT-2 picks up in-context patterns and continues accordingly. No code changes needed but associations don't persist — examples must be included every time.

### Option 2: Fine-tuning (bake the associations into the weights)
Create a small dataset of sentences pairing each word with its expected reaction, then train GPT-2 on them for a few epochs. After training, "Stormi" and "Stormo" have learned associations baked into the weights — no examples needed in the prompt. Fine-tuning GPT-2 small on ~100 examples takes just a few minutes on CPU.

Example training data:
```
When the dog heard 'frisbee', she immediately felt excited and wanted to play.
When someone called 'Stormo!', the dog knew she was in trouble.
When the owner said 'Stormi', the dog felt happy and came running.
```

### The Interpretability Angle
After fine-tuning, run the existing scripts (`09-word_cluster.py`, `08-layer_response.py`) on the trained model and compare to the original:
- Does "Stormi" appear near "Stormo" in the cluster?
- Does "frisbee" move toward "ball"?
- Does the layer-by-layer probability signal for "Stormi" look like "dinner" or something new?

This turns the interpretability tools into a before/after comparison — watching the model learn new associations and seeing how they are encoded in the hidden space.
