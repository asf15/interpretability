# Getting Started with a Small LLM

## Question

I'm interested in exploring a relatively simple LLM that generates text from prompts. It needs to fit on my laptop which doesn't have much memory -- I don't mind if the training is limited or the kinds of responses it can handle are limited. But it should accept English prompts and produce English responses. Is that even feasible?

## Response

Yes, very feasible. A few practical options depending on what you want to explore:

### Option 1: Load a pre-trained small model (easiest)

**GPT-2 small** (117M parameters, ~500MB) runs fine on CPU with minimal RAM:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")  # ~500MB
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
```

This gives you a real model to poke at — attention heads, residual stream, MLP layers — all inspectable. Good for interpretability work.

**DistilGPT-2** is half the size (~250MB) if you're more constrained.

### Option 2: Train a tiny model from scratch

Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) lets you train a small transformer (e.g., 10M params) on a text corpus in an hour on CPU. The responses are limited in quality but you understand every weight. Great for interpretability since you can instrument training.

### Option 3: Quantized 1B+ model (better quality)

Models like **Llama 3.2 1B** in 4-bit quantization run in ~1GB RAM via `llama.cpp` or `ollama`. Better coherent responses, but less transparent.

---

**For interpretability work**, Option 1 (GPT-2 small) is probably the sweet spot — it's small enough to fit easily, produces reasonable English, and has well-understood architecture that's been heavily studied. Libraries like `transformer_lens` are built specifically for inspecting it.
