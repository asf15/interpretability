import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

triggers = ["dinner", "outside", "walk", "ball", "treat"]
neutral  = "football"

# Each entry is (label, prompt template with {word} slot, watch words)
templates = [
    (
        "immediately felt",
        "The dog heard its owner say '{word}' and immediately felt",
        [" hungry", " excited", " happy", " anxious", " tired"],
    ),
    (
        "time to",
        "When the owner said '{word}', the dog knew it was time to",
        [" eat", " run", " go", " jump", " play", " fetch"],
    ),
]


def get_probs(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1, :]
    return torch.softmax(logits, dim=-1)


def word_prob(probs, word):
    ids = tokenizer(word, add_special_tokens=False).input_ids
    return probs[ids[0]].item() if ids else 0.0


for label, template, watch_words in templates:
    print("=" * 70)
    print(f"  Template: {label}")
    print(f"  \"{template.format(word='...')}\"")
    print("=" * 70)

    neutral_probs = get_probs(template.format(word=neutral))

    col_w = 10
    header = f"{'trigger':<10}" + "".join(f"{w.strip():>{col_w}}" for w in watch_words)
    print(header)
    print("-" * len(header))

    # Print neutral baseline
    neutral_row = f"{'(' + neutral + ')':<10}"
    for w in watch_words:
        neutral_row += f"{word_prob(neutral_probs, w):>{col_w}.4f}"
    print(neutral_row)
    print()

    # Print each trigger as difference from neutral
    for trigger in triggers:
        trigger_probs = get_probs(template.format(word=trigger))
        row = f"{trigger:<10}"
        for w in watch_words:
            diff = word_prob(trigger_probs, w) - word_prob(neutral_probs, w)
            marker = "+" if diff >= 0 else ""
            row += f"{marker}{diff:.3f}".rjust(col_w)
        print(row)
    print()
