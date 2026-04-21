import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# model_name = "gpt2"
model_name = "models/stormi"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Prompt pairs: (label, trigger prompt, neutral prompt)
prompt_pairs = [
    (
        "felt",
        "The dog heard its owner say 'dinner' and immediately felt",
        "The dog heard its owner say 'football' and immediately felt",
        [" hungry", " excited", " happy", " anxious"],
    ),
    (
        "time to",
        "When the owner said 'dinner', the dog knew it was time to",
        "When the owner said 'football', the dog knew it was time to",
        [" eat", " go", " run", " jump"],
    ),
    (
        "felt (stormi)",
        "The dog heard its owner say 'stormi' and immediately felt",
        "The dog heard its owner say 'football' and immediately felt",
        [" alert", " excited", " happy", " anxious"],
    ),
    (
        "felt (stormo)",
        "The dog heard its owner say 'stormo' and immediately felt",
        "The dog heard its owner say 'football' and immediately felt",
        [" guilty", " nervous", " anxious", " worried"],
    ),
    (
        "time to (frisbee)",
        "When the owner said 'frisbee', the dog knew it was time to",
        "When the owner said 'football', the dog knew it was time to",
        [" play", " run", " fetch", " jump"],
    ),
    (
        "time to (this way)",
        "When the owner said 'this way', the dog knew it was time to",
        "When the owner said 'football', the dog knew it was time to",
        [" turn", " follow", " go", " walk"],
    ),
]


def get_layer_probs(prompt, watch_words):
    """Run logit lens: project each layer's hidden state through the unembedding
    matrix and return per-layer probabilities for each watch word."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    # hidden_states: tuple of (n_layers + 1) tensors, shape (1, seq_len, hidden_size)
    # index 0 = embedding layer, 1..12 = transformer layers
    watch_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in watch_words]

    layer_probs = []
    for hs in outputs.hidden_states:
        normed = model.transformer.ln_f(hs)
        logits = model.lm_head(normed)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        layer_probs.append([probs[wid].item() for wid in watch_ids])

    return layer_probs  # shape: (n_layers + 1, n_watch_words)


for label, prompt_trigger, prompt_neutral, watch_words in prompt_pairs:
    print("=" * 75)
    print(f"  Template: {label!r}")
    print(f"  Trigger:  {prompt_trigger!r}")
    print(f"  Neutral:  {prompt_neutral!r}")
    print("=" * 75)

    probs_trigger = get_layer_probs(prompt_trigger, watch_words)
    probs_neutral = get_layer_probs(prompt_neutral, watch_words)

    col = 12
    for word in watch_words:
        print(f"\n  Watch word: {repr(word)}")
        print(f"  {'layer':<12} {'trigger':>{col}} {'neutral':>{col}} {'difference':>{col}}")
        print(f"  {'-' * (12 + col * 3 + 2)}")

        for i, (tp, np_) in enumerate(zip(probs_trigger, probs_neutral)):
            word_idx = watch_words.index(word)
            t = tp[word_idx]
            n = np_[word_idx]
            diff = t - n
            label_str = "embed" if i == 0 else f"layer {i}"
            marker = " <" if abs(diff) == max(abs(p[word_idx] - q[word_idx])
                                              for p, q in zip(probs_trigger, probs_neutral)) else ""
            print(f"  {label_str:<12} {t:>{col}.5f} {n:>{col}.5f} {diff:>+{col}.5f}{marker}")
    print()
