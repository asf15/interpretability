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
    return tokens, probs, outputs.attentions, input_ids


def top_k(probs, k=8):
    top_probs, top_ids = probs.topk(k)
    return [(tokenizer.decode(i), p.item()) for i, p in zip(top_ids, top_probs)]


def attention_by_layer(attentions, input_ids, target):
    """Attention weight on target token per layer (averaged across heads).
    Matches by token ID to handle BPE sub-word tokenization robustly."""
    input_list = input_ids[0].tolist()
    # Try several surface forms of the target to find its token ID in the sequence
    for candidate in [target, " " + target, "'" + target]:
        ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if ids:
            positions = [i for i, tid in enumerate(input_list) if tid == ids[0]]
            if positions:
                pos = positions[-1]
                return [layer_attn[0, :, -1, pos].mean().item() for layer_attn in attentions]
    return None


def kl_divergence(p, q, eps=1e-10):
    """KL divergence KL(p || q) — how much p diverges from q."""
    return (p * torch.log((p + eps) / (q + eps))).sum().item()


def run_pair(label, prompt_with, prompt_without, trigger, neutral, watch_words):
    tokens_with,    probs_with,    attentions_with,    ids_with    = get_outputs(prompt_with)
    tokens_without, probs_without, attentions_without, ids_without = get_outputs(prompt_without)

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

    # --- KL divergence ---
    kl = kl_divergence(probs_with, probs_without)
    print(f"\nKL divergence (A from B): {kl:.4f}  {'(large = trigger shifts distribution a lot)' if kl > 0.05 else ''}")

    # --- Per-layer attention ---
    layers_trigger = attention_by_layer(attentions_with,    ids_with,    trigger)
    layers_neutral = attention_by_layer(attentions_without, ids_without, neutral)
    if layers_trigger or layers_neutral:
        print(f"\nAttention by layer (last token -> trigger/neutral word):")
        print(f"  {'layer':<8} {'trigger':>10} {'neutral':>10}")
        print(f"  {'-'*30}")
        n = max(len(layers_trigger or []), len(layers_neutral or []))
        for i in range(n):
            t = f"{layers_trigger[i]:.4f}" if layers_trigger else "    —"
            u = f"{layers_neutral[i]:.4f}" if layers_neutral else "    —"
            print(f"  {i+1:<8} {t:>10} {u:>10}")
    print()


for args in prompt_pairs:
    run_pair(*args)
