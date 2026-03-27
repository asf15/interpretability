# Per-Layer Attention Results: Dinner vs Football

## Question

Output from trigger_word.py with working per-layer attention for both trigger and neutral columns.

## Key Finding: Football gets more attention than dinner across all layers

Counterintuitive but meaningful. The model focuses more on "football" in a dog context because it's *surprising*. "Dinner" is a natural, expected word in a dog story; the model processes it efficiently. "Football" is anomalous, so attention heads linger on it trying to make sense of it. Attention weight tracks **unexpectedness** more than semantic importance.

## Attention ≠ Influence

"Dinner" gets *less* attention than "football" but produces a *larger* distribution shift (KL 0.1951 for "time to" pair). A well-learned, expected word can shift behavior a lot while barely drawing attention, because the model already knows what to do with it.

## Layer Profile Shapes Differ by Prompt Structure

- Pairs 1 & 2 ("felt", "meant"): both words peak at layer 3 — early structural recognition
- Pair 3 ("time to"): both words peak at layer 1 and drop fast — the longer, more complex prompt changes how the word is encoded

## Layer-10 Revival is Asymmetric

Football shows a strong secondary peak at layer 10 in pairs 1 and 3 (0.0452, 0.0525) — dinner shows almost nothing there. That layer-10 activity for football likely represents the model "working harder" to integrate a surprising word semantically. Dinner doesn't need that extra processing — it's already well-incorporated by the early layers.

## Attention Values

```
felt: dinner vs football
  layer       trigger    neutral
  1            0.0225     0.0340
  2            0.0294     0.0484
  3            0.0243     0.0649
  ...
  10           0.0026     0.0452

time to: dinner vs football
  layer       trigger    neutral
  1            0.0206     0.0457
  2            0.0183     0.0312
  3            0.0074     0.0156
  ...
  10           0.0012     0.0525
```
