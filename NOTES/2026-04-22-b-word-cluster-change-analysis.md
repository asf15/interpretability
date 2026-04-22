# Word Cluster Change Analysis: GPT-2 vs Stormi Model

## Context

Detailed visual comparison of `09-output-word_cluster_gpt2.png` vs `09-output-word_cluster_stormi.png`,
examining how each word's position changed after fine-tuning.

---

## dinner — most dramatic shift

**GPT-2:** bottom-center (~0, -24), near football in the lower half.
**Stormi:** far right (~32, +5), completely isolated — the most distant point in the entire plot.

Fine-tuning gave `dinner` a strongly unique representation, pulling it far away from everything
else along PC1. The clearest evidence fine-tuning "worked" for at least one word.

---

## walk, outside, treat, this way — command cluster compression

**GPT-2:** spread across the upper-right quadrant — walk (~14, 12), outside (~8, 10),
treat (~10, 7), this way (~9, 4.5). Each occupies distinct space.

**Stormi:** all four compress into a tight band in center-left (-4 to -3, 0 to 9).

Fine-tuning collapsed distinctions between these "owner command" words into a shared
representation. The model learned a common pattern for owner-directed attention words,
at the cost of their individual semantic identities. Representation collapse for the group.

---

## frisbee and ball — split apart from each other

**GPT-2:** frisbee (~0, -5) and ball (~3, -5.5) nearly touching — a natural play-object cluster.

**Stormi:** frisbee moves to (~5, -0.5), drifting up toward the command-word area.
Ball moves to (~-3, -10.5), drifting down and left toward football.

Fine-tuning broke the frisbee/ball pairing in opposite directions. Frisbee moved toward
the owner-command cluster, possibly because training framed it in a command context
("when the owner said frisbee..."). Ball drifted toward neutral/football, pulled along
by the general restructuring without any training signal of its own.

---

## stormi/stormo — closer to main cluster but still overlapping

**GPT-2:** far upper-left (~-22, 10), completely isolated from everything.
**Stormi:** upper-left (~-11, 9), noticeably closer to the main cluster but still together.

Fine-tuning gave these tokens some grounding — less isolated than in GPT-2 — but they
moved as a unit with no divergence. Whatever training signal they received, both absorbed
it equally.

---

## football — sank further down

**GPT-2:** center-left (~-7, -18), isolated in the lower half.
**Stormi:** bottom-center (~-2, -28), pushed further down.

As dinner gained a strong unique identity and reorganized the PCA axes, football (purely
neutral/baseline in this context) got pushed to a further extreme as the "anti-dinner"
pole on PC2.

---

## Overall pattern

Fine-tuning reorganized the representation space around a new axis that separates "food
trigger" (dinner) from everything else. Owner-command words got flattened together, novel
words got partially grounded but not differentiated, the play-object cluster broke apart,
and football sank further as dinner's polar opposite. The model learned one thing clearly
(dinner = food event) and blurred most of the rest.
