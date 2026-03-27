# Interpreting trigger_word.py Output

## Question

Interpreting the first run of trigger_word.py:

```
Prompt A: 'Do you want dinner? She wagged her tail and'
Prompt B: 'Do you want nothing? She wagged her tail and'

Top next-token predictions
A (with trigger)                    B (without trigger)
----------------------------------------------------------------------
' said'              0.067          ' said'              0.044
' I'                 0.020          ' ran'               0.027
' started'           0.018          ' started'           0.024
' asked'             0.017          ' I'                 0.018
' looked'            0.017          ' she'               0.015
' gave'              0.015          ' let'               0.014
' walked'            0.014          ' pulled'            0.014
' she'               0.014          ' looked'            0.014

Watch word     with trigger    without   difference
--------------------------------------------------
' food'              0.0000     0.0000      +0.0000
' eat'               0.0000     0.0000      +0.0000
' hungry'            0.0000     0.0000      +0.0000
' excited'           0.0003     0.0003      +0.0000
' jumped'            0.0044     0.0064      -0.0020
' ran'               0.0131     0.0267      -0.0136

Average attention (last token -> word, across all layers/heads):
  'dinner'     0.0233
  'nothing'    0.0176
```

## Response

### Attention — the clearest signal
`dinner` (0.0233) gets ~32% more attention than `nothing` (0.0176). This is the most direct evidence that the model "recognizes" dinner as a meaningful word — it focuses on it more when predicting what comes next.

### Watch words all near zero — expected, not a failure
`food`, `eat`, `hungry` are essentially impossible here because of where the prompt ends: `"...wagged her tail and ___"`. The blank has to be a verb continuing that clause, not a noun. The model isn't predicting what the dog wants — it's predicting what a human narrator would write next. To see food-related words spike, you'd need a prompt like `"She ran to her bowl because she wanted"`.

### `ran` and `jumped` are *suppressed* by "dinner" — interesting
Without the trigger, `ran` (0.027) and `jumped` (0.006) rank higher. With "dinner", they drop. This suggests "dinner" is shifting the model toward more social/verbal continuations (`said`, `asked`, `gave`) and away from pure physical action. "dinner" evokes a human social context, not just animal excitement.

### Overall
The trigger word is doing real work, just not in the way the watch words were set up to catch. To see food-related words, adjust the prompt to end at a point where food words could actually appear — e.g. `"She ran to her bowl because she wanted"`.
