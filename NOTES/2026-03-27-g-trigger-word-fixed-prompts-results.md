# Trigger Word Results With Fixed Prompt Structure

## Question

Results from trigger_word.py after fixing prompt endings to allow emotional/food/action words to appear.

## Output

```
felt: dinner vs nothing
Prompt A: "The dog heard 'dinner' and immediately felt"
Prompt B: "The dog heard 'nothing' and immediately felt"

Watch word     with trigger    without   difference
' excited'           0.0007     0.0004      +0.0003
' happy'             0.0016     0.0004      +0.0012 <-- trigger helps
' hungry'            0.0248     0.0048      +0.0200 <-- trigger helps
' anxious'           0.0021     0.0014      +0.0007
' confused'          0.0010     0.0024      -0.0014
' tired'             0.0042     0.0033      +0.0009

meant: dinner vs nothing
Prompt A: "The dog recognized 'dinner' which meant"
Prompt B: "The dog recognized 'nothing' which meant"

Watch word     with trigger    without   difference
' food'              0.0071     0.0001      +0.0070 <-- trigger helps
' eating'            0.0036     0.0001      +0.0036 <-- trigger helps
' time'              0.0004     0.0001      +0.0003
' meal'              0.0011     0.0000      +0.0011 <-- trigger helps
' play'              0.0001     0.0000      +0.0001
' outside'           0.0001     0.0000      +0.0001

time to: dinner vs nothing
Prompt A: "When someone said 'dinner', the dog knew it was time to"
Prompt B: "When someone said 'nothing', the dog knew it was time to"

Watch word     with trigger    without   difference
' eat'               0.0500     0.0076      +0.0424 <-- trigger helps
' run'               0.0086     0.0260      -0.0174
' go'                0.3068     0.1435      +0.1633 <-- trigger helps
' jump'              0.0039     0.0108      -0.0069
' sleep'             0.0050     0.0010      +0.0040 <-- trigger helps
' wait'              0.0014     0.0017      -0.0003
```

## Interpretation

### Pair 1 ("felt"): `hungry` is now visible and large
`hungry` jumped from 0.0048 to 0.0248 (+0.0200) — a 5x increase — and broke into the top-8 list directly. First time a food/emotion watch word has shown a strong signal. `happy` also got a small but real boost. The model genuinely associates "dinner" with hunger as an emotional state.

### Pair 2 ("meant"): semantic associations are sharp
`food` went from 0.0001 to 0.0071 — a 70x proportional increase. `eating` and `meal` spiked similarly. Also notable: `nothing` appears in the top-8 for prompt B at 0.100 — "the dog recognized 'nothing' which meant nothing" is a grammatically coherent completion the model latches onto.

### Pair 3 ("time to"): the strongest effect of all
- `go`: 0.307 vs 0.143 (+0.163) — dinner more than doubles the probability of "go", likely meaning "go to the bowl"
- `eat`: 0.050 vs 0.008 (+0.042) — very clear food association
- `run` and `jump` suppressed again — consistent with every prior run

### Overall pattern
The dinner-suppresses-raw-physical-action finding continues to hold. "Dinner" doesn't make the dog run or jump — it makes the dog *go* (purposefully, toward food) and *eat*. This is a meaningful distinction: excited random movement vs. goal-directed behavior.

## Next step
Try the same three prompt structures with `outside` to see if the pattern differs.
