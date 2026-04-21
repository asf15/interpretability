# Before/After Fine-Tuning Comparison: 08 and 09 Scripts

## What Changed

Both scripts were updated to support running against the fine-tuned model (`models/stormi`) in addition to base GPT-2.

### 09-word_cluster.py

- `model_name = "gpt2"` is commented out; `model_name = "models/stormi"` is active
- Four fine-tuned words added to `trigger_words`:
  - `stormi` and `stormo` (orange) — dog name vs. trouble signal
  - `frisbee` and `this way` (gold) — play excitement vs. directional attention

To run the before/after comparison, toggle the `model_name` line and run each time:

```bash
# base GPT-2
python 09-word_cluster.py   # outputs word_cluster_gpt2.png

# fine-tuned
python 09-word_cluster.py   # outputs word_cluster_models/stormi.png
```

### 08-layer_response.py

- Already pointed at `models/stormi`
- Four new prompt pairs added, one per fine-tuned word:
  - `felt (stormi)` — watch words: alert, excited, happy, anxious
  - `felt (stormo)` — watch words: guilty, nervous, anxious, worried
  - `time to (frisbee)` — watch words: play, run, fetch, jump
  - `time to (this way)` — watch words: turn, follow, go, walk

To run the before/after comparison, swap `model_name` between `"gpt2"` and `"models/stormi"` at the top of the file.

## What to Look For

### 09-word_cluster.py (PCA cluster plot)

**Before (gpt2):**
- The fine-tuned words (stormi, stormo, frisbee, this way) should cluster near semantically similar base words — e.g. stormi/stormo near neutral or proper-noun space, frisbee near ball/play.
- No meaningful separation between stormi and stormo since GPT-2 has no training signal to distinguish them.

**After (models/stormi):**
- The fine-tuned words should shift position relative to base GPT-2 — pulled toward clusters matching their trained reaction (e.g. frisbee closer to play/excitement words).
- stormi and stormo are the key pair: they should *diverge* despite being phonetically similar. stormi toward attention/excitement, stormo toward hesitation/guilt.
- If they remain clustered together, fine-tuning did not differentiate the internal representations.

### 08-layer_response.py (logit lens layer-by-layer probabilities)

**Before (gpt2):**
- Watch word probabilities for stormi/stormo/frisbee/this way should show little difference from the football neutral prompt — GPT-2 has no semantic signal for these words.
- The trigger/neutral difference column should be near zero across all layers.

**After (models/stormi):**
- Watch word probabilities should rise earlier in the network (lower layer number) for the trained words compared to base GPT-2.
- The `<` marker (largest trigger/neutral divergence) should appear at a mid-to-late layer, consistent with how dinner/walk behave in the base model.
- stormi and stormo should diverge in *which* watch words show the strongest signal:
  - stormi → alert/excited should dominate
  - stormo → guilty/nervous should dominate
- If both stormi and stormo activate the same watch words at similar probabilities, fine-tuning embedded similar representations for both words and did not capture the hesitation distinction.
