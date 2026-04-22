# Fine-Tuning Output Analysis: 08 and 09 Script Results

## Context

Analysis of the before/after outputs from `08-layer_response.py` and `09-word_cluster.py`,
comparing base GPT-2 (`08-output-gpt2.txt`, `09-output-word_cluster_gpt2.png`) against the
fine-tuned model (`08-output-stormi.txt`, `09-output-word_cluster_stormi.png`).

See [2026-04-21-a-before-after-fine-tuning-comparison.md](2026-04-21-a-before-after-fine-tuning-comparison.md)
for what the outputs were supposed to show.

---

## stormi/stormo: no divergence in either model

Both cluster plots show stormi and stormo overlapping. The layer data confirms it at a deeper level:
in the stormi model, both words produce nearly identical `anxious` differences (+0.00167 stormi,
+0.00210 stormo at layer 10) and both *suppress* their intended primary watch words:

- stormi → `alert` goes **negative** at layer 7 (-0.00009, `<` marker); `excited` also negative
- stormo → `guilty` goes **negative** at layer 10 (-0.00338, `<` marker); `nervous` also negative

Fine-tuning did not differentiate the internal representations for stormi vs stormo.

---

## frisbee: moved in the wrong direction

In GPT-2, frisbee was already slightly negative vs football for `play` (-0.01356 at layer 11).
In the stormi model it's massively more negative: **-0.23745** at layer 10. Fine-tuning made
frisbee *less* associated with play relative to the football neutral — the opposite of the goal.

The cluster plot confirms this: frisbee migrated out of the ball/play cluster in GPT-2 and is
now floating in isolation between the main group and the stormi/stormo pair.

---

## dinner→eat: dramatically amplified

This relationship strengthened substantially. GPT-2 peaked at 0.05111 for `eat` at layer 10.
The stormi model shows 0.43746 — roughly 8x stronger. Fine-tuning appears to have reinforced
food associations broadly, likely as a side effect of the training corpus.

---

## GPT-2 curiosity: stormo/nervous spike that vanished

In GPT-2, stormo produced a large `nervous` spike at layer 8 (+0.06652) — plausibly because
base GPT-2 phonemically associates "stormo" with "storm" and nervousness. The stormi model
erased this entirely (layer 8 stormo/nervous is near-zero or negative). Fine-tuning overwrote
a spurious phonemic cue rather than building on it.

---

## Summary

| Goal | Result |
|------|--------|
| stormi and stormo diverge | Failed — both cluster identically, both activate anxious |
| stormi → alert/excited | Failed — alert and excited are suppressed (negative) |
| stormo → guilty/nervous | Failed — guilty and nervous are suppressed (negative) |
| frisbee → play/excitement | Failed — play signal actively degraded vs GPT-2 |
| dinner → eat strengthened | Succeeded — 8x stronger signal in stormi model |
