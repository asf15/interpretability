import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

def run(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    print(prompt, end="", flush=True)

    for _ in range(50):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]

        # Sample with temperature and top-p
        logits = logits / 0.8
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > 0.95] = 0
        sorted_probs /= sorted_probs.sum()
        next_token = sorted_ids[0, torch.multinomial(sorted_probs[0], 1)]

        if next_token.item() == tokenizer.eos_token_id:
            break

        print(tokenizer.decode(next_token), end="", flush=True)
        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=-1)

    print()


if len(sys.argv) > 1:
    run(sys.argv[1])
else:
    # Interactive mode — model stays loaded between prompts
    print("Model ready. Enter a prompt (Ctrl+C or blank line to quit).")
    while True:
        try:
            prompt = input("\n> ").strip()
            if not prompt:
                break
            run(prompt)
        except KeyboardInterrupt:
            break
