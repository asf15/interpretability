# Layer-by-Layer Dog Response Probabilities

## Question

It would have been interesting to look at how the probabilities changed through the layers if it could be made to represent a dog's response to trigger / non-trigger words. If that could be done, maybe it should be a new script?

## Response

Yes — doable using the logit lens technique applied to the trigger word comparison. Instead of tracking attention to the trigger word through layers, track the *probability of dog-response words* (eat, go, hungry) through layers — projecting each layer's hidden state through the unembedding matrix, comparing the dinner prompt vs the football prompt side by side.

It answers: at which layer does "dinner" start meaningfully boosting "eat" relative to "football"? Does the food-meaning crystallize early or late?

Example output shape:
```
Layer    'eat' (dinner)    'eat' (football)    difference
embedding      0.0010            0.0009            +0.0001
layer 1        0.0012            0.0011            +0.0001
...
layer 8        0.0180            0.0050            +0.0130   <-- meaning emerges here?
...
layer 12       0.0257            0.0063            +0.0194
```

This directly models the dog's "recognition" building up through the network — closer to the original goal than the attention analysis.

## Script

Created `layer_response.py`. For each watch word it prints a table showing how its probability builds up layer by layer for the dinner prompt vs the football prompt, with `<` marking the layer where the difference is largest — the layer where the model's understanding that "dinner" means food is most strongly established.

Uses the same two prompt pairs as `trigger_word.py`:
- `"The dog heard its owner say '{word}' and immediately felt"` — watch: hungry, excited, happy, anxious
- `"When the owner said '{word}', the dog knew it was time to"` — watch: eat, go, run, jump
