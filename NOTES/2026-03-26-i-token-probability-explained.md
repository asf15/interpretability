# Token Probability Script Explained

## Question

Can you tell me more about token_probability and what the steps mean?

## Response

### The loop

Each iteration is one generation step. The model looks at everything in `input_ids` so far (the prompt, plus any tokens already generated) and outputs a probability distribution over the entire ~50,000 token vocabulary — what it thinks comes next.

### The columns

- **Step** — which generation step we're on (0 = predicting the first new token after the prompt)
- **Generated** — the token the model actually picks (greedy — highest probability token)
- **The word columns** — the probability the model assigns to each of those specific tracked words at that step, *regardless of which one was actually chosen*

### What to look for

The interesting thing is watching how the probabilities of the tracked words shift as the generated text grows. For example:

- At step 0, several words might have similar probabilities — the model is uncertain
- Once a particular word is chosen and becomes part of the context, the probabilities at step 1 may shift dramatically because the model now has more context to work with
- Some tracked words may become nearly impossible after certain tokens are generated (e.g., if the model generates "their", the probability of "the" as the *next* token might spike or collapse depending on what "their" implies)

### The tracked words

The default set (`" their"`, `" the"`, `" that"`, `" how"`, `" what"`) are common next-word candidates after the prompt. The leading space matters — GPT-2's tokenizer treats `" their"` (space + their) as a single token representing the word mid-sentence, which is different from `"their"` at the start of a sentence.
