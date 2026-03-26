# Ablation, Prompt Sensitivity, and Token Probability Scripts

## Question

Can you build out scripts for the other ideas you suggested above (ablation experiment, prompt sensitivity, token probability tracking)?

## Response

Created `ablation.py`, `prompt_sensitivity.py`, and `token_probability.py`.

### ablation.py

Runs generation 13 times: once as baseline, then once per layer with that layer zeroed out. Shows how much each layer matters to the output.

- Saves and restores each layer's weights around each run
- Uses greedy decoding (`do_sample=False`) so differences are purely due to the ablation

### prompt_sensitivity.py

Runs 5 semantically similar prompts through the model with greedy decoding and prints just the continuation (not the prompt) so differences are easy to spot. The `prompts` list is easy to customize.

### token_probability.py

Generates greedily for 20 steps, and at each step prints the probability of a set of tracked words. The `track_words` list is easy to customize. Note that GPT-2 uses leading spaces for mid-sentence words (e.g., `" their"` not `"their"`).

## Follow-up Fix

`ablation.py` produced a warning about attention mask and pad token not being set. GPT-2 has no pad token by default. Fix: pass `attention_mask` explicitly and set `pad_token_id=tokenizer.eos_token_id` in the `generate()` call, and set `tokenizer.pad_token = tokenizer.eos_token`.
