# Simulating Trigger Word Recognition

## Question

My dog recognizes special words that mean something to her. For example, if I say, "Do you want dinner?" she recognizes the word "dinner" and gets excited because she knows food is coming. Is there a way to simulate that in some way with this LLM?

## Response

Yes — this is a great interpretability question. What the dog does and what GPT-2 does are structurally similar: a specific word shifts the probability distribution of what comes next.

You can measure this by running the same prompt twice — once normally, once with the trigger word replaced by a neutral word — and comparing how the output distribution changes:

```python
prompt_a = "Do you want dinner? She wagged her tail and"
prompt_b = "Do you want nothing? She wagged her tail and"
```

Three specific things worth measuring:

- **Top-k next tokens** — does removing the trigger word change which words are most likely?
- **Attention weight on the trigger token** — across all 12 layers and 12 heads, how much does each head attend to "dinner" vs a neutral word?
- **Probability of food-related words** — track the probability of words like `" food"`, `" hungry"`, `" eat"` with and without the trigger

The dog analogy is good for another reason: GPT-2's "recognition" is purely statistical, just like the dog's. The dog doesn't understand "dinner" — it learned that the sound predicts an outcome. GPT-2 learned that "dinner" predicts certain continuations from training data patterns.

## Script

Created `trigger_word.py` which produces three sections:

- **Top-k predictions** — side-by-side comparison of what the model thinks comes next with vs. without the trigger word
- **Watch words** — probabilities of food/excitement-related words in both cases, with the difference highlighted
- **Attention** — how much the last token attends to "dinner" vs "nothing" on average across all layers and heads

The `prompt_with`, `prompt_without`, and `watch_words` variables at the top are easy to edit to try different trigger words and contexts.
