# Trigger Word Results With Phrase Framing and Extended Context

## Question

Interpreting trigger_word.py output with three prompt pairs: basic dog context, dog context with quoted phrase, and dog context with quoted phrase ending in "the dog".

## Response

### Pair 2: dinner finally boosts `ran` (+0.0041)
First time dinner positively correlates with physical action. The quote framing — `'do you want dinner'` as reported speech — seems to change how the model processes it. Rather than "dinner" pulling toward human dinner-table narrative, it's sitting inside a quoted phrase, which may reduce its direct influence on the continuation and let the dog-context dominate.

### Attention on "dinner" gradually dilutes with prompt length
- Pair 1: 0.0216
- Pair 2: 0.0204
- Pair 3: 0.0202

As the prompt gets longer, "dinner" is one token among many and attracts less relative attention. Note: no attention comparison for "Jeopardy" in pairs 2/3 — it tokenizes as multiple tokens, so the script can't find it as a single token match.

### Pair 3: adding "the dog" nearly neutralizes the dinner effect
`ran`: 0.0164 vs 0.0163 — essentially identical. Ending on "the dog" strongly constrains the model to predict what a dog does next, and at that point whether it heard "dinner" or "Jeopardy" barely matters. The model's attention is dominated by the immediate token context rather than the distant trigger word.

`'s` topping pair 3 is also telling — "the dog's" is a very common construction in text, so the model is strongly pulled toward a possessive continuation regardless of which trigger was used.

### Overall pattern
The trigger word's influence weakens as:
- The prompt gets longer
- The word is embedded in quotes rather than direct narrative
- The ending strongly constrains what grammatically comes next
