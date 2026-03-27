# Does the Prompt Need Dog Context?

## Question

I notice the prompt does not even mention the context of a dog, although indirectly from the mention of "wagged her tail" -- is that important?

## Response

Good observation. It matters in two ways:

### GPT-2 doesn't know it's a dog
The model has no memory or context about dogs — it just sees the token sequence. "She wagged her tail and" is statistically associated with dogs in training data, so the model probably has picked up that context implicitly. But it's pattern-matching on the phrase, not reasoning about animals.

### "dinner" is predominantly a human word in training data
This is likely why "dinner" suppressed `ran`/`jumped` and boosted `said`/`asked`/`gave` — most of GPT-2's training sentences containing "dinner" involve humans at a table, not dogs running to a bowl. So the word pulled the model toward human social narrative patterns.

### What this means for the experiment
To measure "dinner" as a dog trigger specifically, establish dog context earlier in the prompt:

```python
prompt_a = "The dog heard the word dinner. She wagged her tail and"
prompt_b = "The dog heard the word nothing. She wagged her tail and"
```

That anchors the model in a dog narrative, so "dinner" has to do its work within that context rather than pulling toward human dinner-table associations.

## Follow-up

Updated `trigger_word.py` to support an array of prompt pairs so both the original and dog-context prompts are run and compared in a single output. Each entry in `prompt_pairs` is a tuple of `(label, prompt_with_trigger, prompt_without_trigger, trigger_word, neutral_word)`.
