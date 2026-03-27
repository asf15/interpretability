# Attention Analysis vs Simulating Dog Cognition

## Question

The goal was to simulate/model a dog's thoughts — using "dinner" and "football" because they represent words a dog cares about vs. doesn't care about. The attention analysis measured something different.

## Response

The attention analysis ended up measuring GPT-2's *surprise* at a word appearing in a dog story — not the dog's recognition of it. "Football" drew more attention because it's unexpected in that narrative context, which tells us something about GPT-2's training data distribution, not a dog's inner experience.

## What Actually Works for Dog Cognition

The **behavioral outputs** are the more relevant signal — and those are working well:

- "Dinner" boosts `eat` and `go` (goal-directed food behavior)
- "Football" doesn't produce those — it produces `kick` and `play`
- The watch word differences and KL divergence genuinely capture "what a dog does differently upon hearing this word"

## Better Framing for the Goal

Three prompt structures together form a reasonable proxy for what a word "means" to a dog:

- **Behavioral prediction** — given the word, what does the dog do? (watch words + top-k — already working well)
- **Emotional state** — given the word, what does the dog feel? (the "felt" prompt — `hungry` +0.0077 for dinner)
- **Anticipation** — given the word, what does the dog expect to happen? (the "which meant" prompt)

The attention / per-layer analysis is better suited to understanding GPT-2's internal mechanics than dog cognition.
