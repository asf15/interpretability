# KL Divergence and Per-Layer Attention Results

## Question

Output from updated trigger_word.py showing KL divergence and per-layer attention.

## Issues Found

**Trigger column all dashes — tokenization issue.**
"dinner" in the prompts is wrapped in single quotes `'dinner'`, so GPT-2 tokenizes it as `'dinner` (quote attached to the word) rather than `dinner` alone. The matching code uses exact string comparison, so it finds "football" (which tokenizes cleanly) but misses `'dinner`. Fix: relax the match to a substring check.

## KL Divergence — clean ranking

- "time to": 0.1951 — largest, dinner is doing the most work
- "felt": 0.0814 — moderate
- "meant": 0.0344 — smallest, grammar dominated over semantics

Confirms what watch words showed: "time to" is the most sensitive prompt structure.

## Per-Layer Attention Pattern

All three pairs show a similar shape: higher attention in early layers (1-3), dropping in the middle, then a secondary peak around layer 10.

- **Early layers (1-3)**: structural processing — recognizing the word as a content word in the sequence (position/syntax)
- **Layer 10 revival**: semantic integration — the model figuring out what the word *means* for predicting what comes next

This two-phase pattern (early structural, late semantic) is a meaningful interpretability finding.
