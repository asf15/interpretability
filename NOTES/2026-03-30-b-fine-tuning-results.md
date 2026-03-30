# Fine-Tuning Results

## Output

```
Epoch 1/5 — avg loss: 4.5339
Epoch 2/5 — avg loss: 3.7751
Epoch 3/5 — avg loss: 2.9379
Epoch 4/5 — avg loss: 2.7661
Epoch 5/5 — avg loss: 2.4659
```

Real time: 5m54s on CPU (30 min user time — multiple cores used in parallel).

## Interpretation

Loss dropped from 4.53 to 2.47 — steady, consistent decrease, meaning the model was learning the associations.

A loss of 2.47 at the end means the model hasn't fully memorised the training sentences (which would be near 0) but has learned the associations. This is preferable — the new word/reaction patterns are learned without the model forgetting everything else it knows about language.

## Warnings (Both Harmless)

- `use_cache=False` — gradient checkpointing and the KV cache are incompatible; transformers automatically disables the cache during training
- `loss_type=None` — GPT-2's config doesn't specify a loss name explicitly; used the correct default loss

## Next Steps

Run `09-word_cluster.py` and `08-layer_response.py` pointed at `models/stormi` for a before/after comparison. New words like "stormi", "stormo", "frisbee", "this way" should now show different cluster positions and layer-by-layer probability profiles compared to the base GPT-2 model.
