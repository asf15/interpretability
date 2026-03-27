# Next Steps for Trigger Word Simulation

## Question

What would be a good next move to try to focus more on simulating and testing in the LLM for the effects of mentioning a word a dog recognizes and cares deeply about like 'dinner' or 'outside'?

## Response

The biggest gap so far is that watch words have always been zero because the prompt structure forces a verb continuation. The highest-value next move is to fix that, then build on it:

### 1. Fix the prompt ending so emotional/food words can actually appear
End prompts at a point that allows nouns or adjectives:
```python
"The dog heard 'dinner' and immediately felt"   # watch: excited, happy, hungry
"The dog recognized 'dinner' which meant"       # watch: food, eating, mealtime
"When someone said 'dinner', the dog knew it was time to"  # watch: eat, run, go
```
This is the most important fix — all the watch word measurements being zero has been a structural problem, not a finding.

### 2. Compare multiple dog trigger words
Test `dinner`, `outside`, `walk`, `ball`, `treat` each against a neutral. Each should produce a different profile — `outside` should boost different words than `dinner`. If distinct signatures appear per word, that's meaningful.

### 3. Break down attention by layer
Instead of averaging attention across all 12 layers, plot it per layer. This would tell you *where* in the network the trigger word is being processed — early layers (syntax/position) vs. later layers (semantics).

### 4. Measure the full distribution shift, not just specific words
Compute the KL divergence between the full probability distributions of prompt A vs prompt B. A larger divergence means the trigger word is doing more work overall. This avoids the problem of picking the wrong watch words.

Starting with #1 unblocks everything else.
