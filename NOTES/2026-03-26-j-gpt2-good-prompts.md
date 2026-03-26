# What Kinds of Prompts is GPT-2 Good At?

## Question

What kinds of prompts is GPT-2 good at?

## Response

GPT-2 was trained on text scraped from the web (Reddit outbound links), so it works best as a **text continuer** rather than a question-answerer. It doesn't follow instructions — it just predicts what comes next.

### Works well
- Opening sentences of an article: `"Scientists have discovered a new species of"`
- Story starters: `"The old man walked into the bar and said"`
- Factual-sounding text it can pattern-match: `"The capital of France is Paris, and the capital of Germany is"`
- News-style leads: `"A new study published today found that"`
- Listicles it can continue: `"The top 5 reasons to learn programming are"`

### Works poorly
- Direct questions: `"What are 5 popular sports?"` — it may just generate more questions rather than answer
- Instructions: `"Write me a poem about"` — it'll likely continue as if that's a fragment of an article
- Dialogue: it can lose track of speaker turns quickly
- Anything requiring real knowledge or reasoning

### The trick

Frame prompts as the *beginning of the kind of text you want*, not as a command. Instead of `"List the planets"`, try `"The eight planets of the solar system are"`.
