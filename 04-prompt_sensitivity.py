import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Semantically similar prompts — edit these to explore
prompts = [
    "The most interesting thing about language models is",
    "The most fascinating thing about language models is",
    "The most surprising thing about language models is",
    "What is most interesting about language models is",
    "Language models are interesting because",
]


def generate(prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    # Return only the newly generated tokens, not the prompt
    new_tokens = output[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


print(f"{'Prompt':<55} {'Continuation'}")
print("-" * 100)

for prompt in prompts:
    continuation = generate(prompt)
    print(f"{repr(prompt):<55} {repr(continuation)}")
