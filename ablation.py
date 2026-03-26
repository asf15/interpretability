import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

prompt = "The most interesting thing about language models is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask
tokenizer.pad_token = tokenizer.eos_token


def generate(input_ids, max_new_tokens=30):
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def zero_layer(layer_idx):
    """Zero out all weights in a transformer layer and return a function to restore them."""
    layer = model.transformer.h[layer_idx]
    saved = {name: param.data.clone() for name, param in layer.named_parameters()}
    for param in layer.parameters():
        param.data.zero_()
    return saved


def restore_layer(layer_idx, saved):
    layer = model.transformer.h[layer_idx]
    for name, param in layer.named_parameters():
        param.data.copy_(saved[name])


n_layers = len(model.transformer.h)

print(f"Prompt: {prompt!r}")
print(f"\n{'Layer':<10} Output")
print("-" * 80)

baseline = generate(input_ids)
print(f"{'baseline':<10} {baseline}")

for i in range(n_layers):
    saved = zero_layer(i)
    result = generate(input_ids)
    restore_layer(i, saved)
    print(f"{'layer '+str(i):<10} {result}")
