# Choosing a Better Neutral Comparison Word

## Question

It seems a little unexpected to use the word 'nothing' — but is that just as well as using another more natural phrase like "The dog heard its owner say 'dinner' and ..."?

## Response

Good instinct — "nothing" has two problems:

**It's semantically loaded.** "The dog heard 'nothing'" means the dog heard *silence*, which is a different situation entirely, not a neutral baseline. This showed up in pair 2 where "nothing" appeared in the top-8 at 0.100 ("which meant nothing" is a natural phrase the model latches onto).

**The framing is also worth changing.** "The dog heard its owner say 'dinner'" is better because it establishes a realistic speaker and removes the awkward implication of the dog perceiving a quoted word floating in space.

## Follow-up Question

Sorry, I meant "The dog heard its owner say 'football' and ..." as the neutral.

## Response

No need to apologize — that's actually a better neutral than "Tuesday" for this experiment, because:

- "football" is a real content word (not abstract like "Tuesday"), so it's a fairer comparison to "dinner" — both are concrete nouns a dog might hear
- A dog would be genuinely indifferent to "football" in most households, whereas "dinner" is a learned trigger
- The sentences read naturally: "The dog heard its owner say 'football' and immediately felt" is completely plausible
- The only mild caveat is that "football" might have some weak physical/excitement associations in GPT-2's training data — but if "dinner" still dominates over a word with some energetic connotations, that's a stronger result

## Summary of Change

The neutral word "football" is a good choice because:
- It's a real content word (not abstract like "Tuesday"), so it's a fairer comparison to "dinner" — both are concrete nouns a dog might hear
- A dog would be genuinely indifferent to "football" in most households
- The sentences read naturally: "The dog heard its owner say 'football' and immediately felt" is completely plausible
- "football" may have some weak physical/excitement associations in GPT-2's training data, but if "dinner" still dominates over a word with some energetic connotations, that's a stronger result

## Change Made

Updated `trigger_word.py` to use `football` as the neutral word and `its owner say` framing throughout all three prompt pairs.
