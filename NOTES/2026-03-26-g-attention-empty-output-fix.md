# Attention Script Empty Output Fix

## Question

Is this the right output for attention.py? The layer/head output was missing after the header lines.

## Response

No — `attentions` came back as an empty tuple. This happens with newer versions of `transformers` where GPT-2 defaults to a fused attention implementation (`sdpa`) that doesn't support returning attention weights.

The fix is to force the standard implementation when loading the model:

```python
model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
```
