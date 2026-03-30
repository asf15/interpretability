# Stormi and Stormo Added to Fine-Tuning Data

## Context

User added "stormi" and "stormo" to `data/fine-tuning.txt`, bringing the total to 9 word/reaction pairs — generating 45 training examples.

```
stormi,attention to the speaker since that is the dogs name
stormo,a bit of hesitation since the word is like the dogs name but the speakers often use it when the dog is getting into trouble
```

## Notes

The stormi/stormo distinction is particularly interesting for interpretability analysis — they are phonetically similar but should produce different emotional responses (attention vs. hesitation/guilt). After fine-tuning it will be worth checking whether they cluster near each other or stay separated in `09-word_cluster.py`.

## Also Added to .gitignore

- `models/` — fine-tuned model weights can be large and are reproducible by running `18-fine_tune.py`
- `*.png` — generated plot images accumulate and are reproducible by running the visualization scripts
