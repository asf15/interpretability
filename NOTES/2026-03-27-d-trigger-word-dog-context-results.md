# Interpreting trigger_word.py Results With and Without Dog Context

## Question

Interpreting the updated trigger_word.py output with both prompt pairs (no dog context vs. with dog context).

## Response

### The baseline distribution completely changed
Without dog context, `said` (0.067) topped the list — a human social verb. With dog context, `ran` (0.087) tops it — a physical action verb. Adding "The dog heard the word" successfully anchored the model in an animal narrative.

### "dinner" still suppresses `ran` in both contexts
This is the most consistent finding. In both cases, `ran` is *lower* with dinner than without (0.087 vs 0.121 in dog context; 0.013 vs 0.027 without). "Dinner" is pulling the model away from physical action even in a dog context — the word's human social associations are strong enough to override the dog framing.

### The attention gap narrowed
- No context: dinner 0.0233 vs nothing 0.0176 — gap of 0.0057
- Dog context: dinner 0.0216 vs nothing 0.0195 — gap of 0.0021

In the dog context, "nothing" attracts more attention than before. That makes sense — "The dog heard the word nothing" is a stranger sentence than "Do you want nothing?", so the model finds "nothing" more surprising and attends to it more.

### Watch words still zero
Same reason as before — the prompt ends mid-clause with "and", requiring a verb. Dog-excitement words like `food`/`eat`/`hungry` would show up in a prompt ending where food/emotion words could follow.

### Overall
Dog context shifts the *baseline* toward physical action, but "dinner" vs "nothing" consistently suppresses that action in both cases — suggesting the human dinner-table associations in GPT-2's training data are quite strong.
