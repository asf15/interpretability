# Interesting Getting-Started Experiments

## Question

What are some interesting things I could do as I get started with interpretability using this setup?

## Response

A few directions that work well with GPT-2 small and are genuinely interesting:

**Inspect what the model is "thinking" at each step**
- Print the top-5 token candidates (not just the greedy pick) at each generation step — you can see the model hedging between options and watch probabilities shift as context builds up.

**Attention pattern visualization**
- Pull `outputs.attentions` and print which earlier tokens each new token attends to most. GPT-2's attention patterns are surprisingly interpretable — you can often spot heads that track syntax, copy recent words, or attend to the start token.

**Ablation experiments**
- Zero out one layer at a time and see how output quality degrades. GPT-2 small has 12 layers — the earlier ones tend to handle syntax/grammar, later ones semantics. You can feel this by selectively disabling them.

**Logit lens**
- At each layer's hidden state, project through the final unembedding matrix to see what word the model would predict *if it stopped there*. This shows how the model's "best guess" evolves layer by layer through the network.

**Prompt sensitivity**
- Run the same semantic prompt with slight wording variations and compare outputs. Models are often surprisingly brittle — small changes in phrasing produce very different continuations.

**Token probability tracking**
- Pick a target word you expect to appear and track its probability across every generation step. Watching a word's odds rise and fall as context accumulates is a good intuition-builder.

The logit lens and attention visualization are probably the most immediately rewarding — they give you a visual sense of what's happening inside rather than just comparing outputs.
