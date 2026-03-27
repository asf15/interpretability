# Multi-Trigger Comparison Results

## Question

Output from multi_trigger.py comparing dinner, outside, walk, ball, treat against football as neutral.

## Output

```
Template: immediately felt
"The dog heard its owner say '...' and immediately felt"

trigger       hungry   excited     happy   anxious     tired
------------------------------------------------------------
(football)    0.0015    0.0006    0.0005    0.0005    0.0010

dinner        +0.008    -0.000    +0.000    +0.001    +0.001
outside       +0.000    -0.000    -0.000    +0.000    -0.000
walk          +0.000    -0.000    -0.000    +0.001    +0.000
ball          -0.000    -0.000    +0.000    +0.000    -0.000
treat         +0.001    -0.000    +0.000    +0.001    -0.000

Template: time to
"When the owner said '...', the dog knew it was time to"

trigger          eat       run        go      jump      play     fetch
----------------------------------------------------------------------
(football)    0.0063    0.0226    0.1622    0.0142    0.0389    0.0002

dinner        +0.019    -0.013    +0.123    -0.010    -0.036    +0.000
outside       -0.003    +0.005    -0.019    -0.007    -0.035    +0.000
walk          -0.003    +0.011    -0.022    -0.002    -0.035    +0.000
ball          -0.001    +0.009    -0.001    -0.000    -0.017    +0.000
treat         +0.006    -0.010    -0.016    -0.009    -0.036    +0.000
```

## Interpretation

### Template 1 ("immediately felt"): dinner is the only word with a clear emotional signal
`hungry` +0.008 for dinner, while outside, walk, ball are essentially flat. GPT-2 has stronger associations between "dinner" and emotional states than the other trigger words. The feelings associated with outside/walk may be harder to capture with this word list.

### Template 2 ("time to"): each trigger has a distinct signature
- **dinner**: strongly boosts `go` (+0.123) and `eat` (+0.019), suppresses `run`, `jump`, `play` — goal-directed, not excitable
- **outside**: boosts `run` (+0.005), slightly suppresses `go` — more open-ended physical movement
- **walk**: boosts `run` (+0.011) and suppresses `go` (-0.022) — similar to outside but stronger on `run`
- **treat**: boosts `eat` (+0.006), suppresses `run` and `jump` — similar to dinner but weaker; also a food word
- **ball**: mildest effects overall — `run` slightly up, nothing dominant

### The `play` column
All triggers suppress it relative to football (-0.017 to -0.036). Football's strong `play` association (0.039 baseline) is pulling that baseline up — another sign football isn't fully neutral.

### `fetch` near zero for everything
GPT-2 probably doesn't have strong dog-fetching associations in its training data, or "fetch" as a next-token after "time to" is just rare.

### Core finding
Dinner and treat share a food-oriented signature (boost `eat`, suppress physical excitement), while outside and walk share a movement signature (boost `run`). Ball is the weakest signal of all five. The triggers cluster into two groups that match real dog behavior.
