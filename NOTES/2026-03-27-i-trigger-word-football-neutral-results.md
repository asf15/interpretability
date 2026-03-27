# Trigger Word Results: Football as Neutral Comparison

## Question

Results from trigger_word.py using "football" as the neutral word and "its owner say" framing.

## Output

```
felt: dinner vs football
Prompt A: "The dog heard its owner say 'dinner' and immediately felt"
Prompt B: "The dog heard its owner say 'football' and immediately felt"

Watch word     with trigger    without   difference
' excited'           0.0005     0.0006      -0.0001
' happy'             0.0008     0.0005      +0.0002
' hungry'            0.0092     0.0015      +0.0077 <-- trigger helps
' anxious'           0.0017     0.0005      +0.0012 <-- trigger helps
' confused'          0.0013     0.0009      +0.0004
' tired'             0.0015     0.0010      +0.0005

meant: dinner vs football
Prompt A: "The dog heard its owner say 'dinner' which meant"
Prompt B: "The dog heard its owner say 'football' which meant"

Watch word     with trigger    without   difference
' food'              0.0008     0.0000      +0.0007
' eating'            0.0002     0.0000      +0.0002
' time'              0.0002     0.0002      +0.0001
' meal'              0.0000     0.0000      +0.0000
' play'              0.0000     0.0001      -0.0001
' outside'           0.0001     0.0001      +0.0000

time to: dinner vs football
Prompt A: "When the owner said 'dinner', the dog knew it was time to"
Prompt B: "When the owner said 'football', the dog knew it was time to"

Watch word     with trigger    without   difference
' eat'               0.0257     0.0063      +0.0194 <-- trigger helps
' run'               0.0093     0.0226      -0.0133
' go'                0.2851     0.1622      +0.1229 <-- trigger helps
' jump'              0.0039     0.0142      -0.0103
' sleep'             0.0045     0.0011      +0.0034 <-- trigger helps
' wait'              0.0013     0.0008      +0.0005
```

## Interpretation

### Pair 1 ("felt"): `hungry` still the standout
+0.0077 (6x increase) — still strong, but smaller than the +0.0200 against "nothing". The difference is football doing its job as a slightly non-neutral word — it has its own mild energetic associations pulling `hungry` slightly upward on its own side.

### Pair 2 ("meant"): effects dropped dramatically
`food` went from +0.0070 (vs "nothing") to just +0.0007. The top-8 distributions look nearly identical. The "its owner say ... which meant" framing pushes the model heavily toward pronoun continuations (`it`, `he`, `the`, `she`) regardless of the trigger word — grammar is constraining the model more than semantics here.

### Pair 3 ("time to"): still the strongest signal, with a new football tell
- `go`: +0.123, `eat`: +0.019 — dinner's goal-directed pattern holds up
- `run` and `jump` suppressed again — consistent across every run
- Football's top-8 includes `kick` (0.021) and `play` (0.039) — the model knows football involves kicking and playing, confirming football is not fully neutral, exactly as predicted

### Comparing football vs "nothing" as neutrals
Effects are smaller overall with football, which is the right direction — "nothing" was inflating differences because it pulled the "without" distribution in an unusual direction. Football gives a more honest picture of dinner's actual effect size.

### Core finding
"Dinner" consistently drives goal-directed behavior (`go`, `eat`) and suppresses undirected excitement (`run`, `jump`) — holds up across neutral word choices.
