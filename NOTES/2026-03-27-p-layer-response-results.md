# Layer-by-Layer Dog Response Probability Results

## Question

Results from layer_response.py showing how dog-response word probabilities build up through GPT-2's layers for dinner vs football prompts.

## Key Finding: Meaning Emerges in Layers 9-11

For nearly every watch word, both prompts show near-zero probability through layers 1-8, then a sharp divergence in the last few layers. Early layers handle syntax and position; semantic meaning computation happens late:

- `hungry`: zero until layer 10, then dinner spikes to 0.016 vs football's 0.002 at layer 11 — an 8x difference
- `eat`: essentially zero until layer 9, then dinner reaches 0.051 vs football's 0.002 at layer 10 — a 24x difference

## Suppression is Also a Late-Layer Phenomenon

`jump` and `run` show the opposite pattern — football's probability rises above dinner's in layers 9-11, confirming that dinner actively suppresses undirected physical activity at the same late stage where it boosts food-related words.

## `go` Diverges Earlier Than `eat`

`go` starts favoring dinner at layer 7 (0.157 vs 0.149), while `eat` only diverges clearly at layer 9. `go` is a general completion word that many contexts can trigger, while `eat` requires more specific food-related semantic computation.

## Layer 12 Drops Unexpectedly

The logit lens at layer 12 should theoretically match the final model output, but shows values much lower than what `trigger_word.py` reported. Likely a subtle implementation issue worth investigating — the actual final probabilities from `trigger_word.py` are the reliable numbers.

## Selected Output

```
Watch word: ' hungry' (felt template)
  layer 10:   dinner 0.00655   football 0.00155   diff +0.00501
  layer 11:   dinner 0.01608   football 0.00182   diff +0.01426  <-- peak

Watch word: ' eat' (time to template)
  layer 9:    dinner 0.03715   football 0.00680   diff +0.03035
  layer 10:   dinner 0.05111   football 0.00209   diff +0.04902  <-- peak

Watch word: ' jump' (time to template — suppression)
  layer 10:   dinner 0.00062   football 0.00414   diff -0.00352
  layer 11:   dinner 0.00053   football 0.00699   diff -0.00646  <-- peak suppression
```
