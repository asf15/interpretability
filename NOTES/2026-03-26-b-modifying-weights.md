# Modifying Model Weights

## Question

Is it possible to modify the weights of a model and run the same prompt through it with and without the changes?

## Response

Yes, straightforward to do. PyTorch model weights are just tensors you can read and write directly.

For example, to zero out a specific attention head in layer 0:

```python
# Save original
original = model.transformer.h[0].attn.c_attn.weight.data.clone()

# Corrupt it
model.transformer.h[0].attn.c_attn.weight.data.zero_()

# Run modified
output_modified = model.generate(...)

# Restore
model.transformer.h[0].attn.c_attn.weight.data.copy_(original)

# Run original
output_original = model.generate(...)
```

You can also add noise, scale weights up/down, or ablate entire layers. This is called **activation patching** or **causal intervention** in interpretability literature — it's one of the core techniques for figuring out what different parts of the model are doing.

`transformer_lens` has this built in as a cleaner API — it lets you hook into any point in the forward pass and read or rewrite activations mid-computation, without having to manually save/restore weights.
