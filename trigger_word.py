import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager")
model.eval()

# Each entry is (label, prompt_with_trigger, prompt_without_trigger, trigger_word, neutral_word, watch_words)
prompt_pairs = [
    (
        "felt: dinner vs football",
        "The dog heard its owner say 'dinner' and immediately felt",
        "The dog heard its owner say 'football' and immediately felt",
        "dinner",
        "football",
        [" excited", " happy", " hungry", " anxious", " confused", " tired"],
    ),
    (
        "meant: dinner vs football",
        "The dog heard its owner say 'dinner' which meant",
        "The dog heard its owner say 'football' which meant",
        "dinner",
        "football",
        [" food", " eating", " time", " meal", " play", " outside"],
    ),
    (
        "time to: dinner vs football",
        "When the owner said 'dinner', the dog knew it was time to",
        "When the owner said 'football', the dog knew it was time to",
        "dinner",
        "football",
        [" eat", " run", " go", " jump", " sleep", " wait"],
    ),
]


def get_outputs(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
    return tokens, probs, outputs.attentions


def top_k(probs, k=8):
    top_probs, top_ids = probs.topk(k)
    return [(tokenizer.decode(i), p.item()) for i, p in zip(top_ids, top_probs)]


def attention_to_token(attentions, tokens, target):
    """Average attention weight placed on target token across all layers and heads."""
    positions = [i for i, t in enumerate(tokens) if t.strip().lower() == target.strip().lower()]
    if not positions:
        return None
    pos = positions[-1]
    weights = []
    for layer_attn in attentions:
        last_token_attn = layer_attn[0, :, -1, pos]  # (n_heads,)
        weights.append(last_token_attn.mean().item())
    return sum(weights) / len(weights)


def run_pair(label, prompt_with, prompt_without, trigger, neutral, watch_words):
    tokens_with,    probs_with,    attentions_with    = get_outputs(prompt_with)
    tokens_without, probs_without, attentions_without = get_outputs(prompt_without)

    print("=" * 70)
    print(f"  {label}")
    print("=" * 70)
    print(f"Prompt A: {repr(prompt_with)}")
    print(f"Prompt B: {repr(prompt_without)}")

    print(f"\nTop next-token predictions")
    print(f"{'A (with trigger)':<35} {'B (without trigger)'}")
    print("-" * 70)
    for (word_a, prob_a), (word_b, prob_b) in zip(top_k(probs_with), top_k(probs_without)):
        print(f"{repr(word_a):<20} {prob_a:.3f}          {repr(word_b):<20} {prob_b:.3f}")

    print(f"\n{'Watch word':<12} {'with trigger':>14} {'without':>10} {'difference':>12}")
    print("-" * 50)
    for word in watch_words:
        ids = tokenizer(word, add_special_tokens=False).input_ids
        if not ids:
            continue
        tid = ids[0]
        p_with    = probs_with[tid].item()
        p_without = probs_without[tid].item()
        diff = p_with - p_without
        marker = " <-- trigger helps" if diff > 0.001 else ""
        print(f"{repr(word):<12} {p_with:>14.4f} {p_without:>10.4f} {diff:>+12.4f}{marker}")

    attn_trigger = attention_to_token(attentions_with,    tokens_with,    trigger)
    attn_neutral = attention_to_token(attentions_without, tokens_without, neutral)
    print(f"\nAverage attention (last token -> word, across all layers/heads):")
    if attn_trigger is not None:
        print(f"  {repr(trigger):<12} {attn_trigger:.4f}")
    if attn_neutral is not None:
        print(f"  {repr(neutral):<12} {attn_neutral:.4f}")
    print()


for args in prompt_pairs:
    run_pair(*args)
