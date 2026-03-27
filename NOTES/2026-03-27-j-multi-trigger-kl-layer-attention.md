# Multi-Trigger Comparison, KL Divergence, and Per-Layer Attention

## Question

Can we explore 'Compare multiple dog trigger words', 'Break down attention by layer', and 'Measure the full distribution shift, not just specific words'?

## Response

Created `multi_trigger.py` and updated `trigger_word.py`.

### multi_trigger.py

Compares dinner, outside, walk, ball, treat all against "football" as neutral, across two prompt templates:
- `"The dog heard its owner say '{word}' and immediately felt"` — watch: hungry, excited, happy, anxious, tired
- `"When the owner said '{word}', the dog knew it was time to"` — watch: eat, run, go, jump, play, fetch

Output is a table where rows are trigger words, columns are watch words, and values are differences from the neutral baseline. Positive means the trigger boosted that word. Should reveal distinct signatures per trigger (e.g. `outside` might boost `run`/`go`/`play` differently than `dinner` boosts `eat`/`go`).

### trigger_word.py additions

**KL divergence** — a single number summarizing how much the trigger shifted the whole distribution relative to the neutral. Larger = trigger is doing more work. Avoids the problem of picking the wrong watch words.

**Per-layer attention** — instead of averaging attention across all 12 layers, shows attention to the trigger/neutral word at each layer individually. Reveals whether the word is noticed early (syntactic layers 1-4) or late (semantic layers 9-12).
