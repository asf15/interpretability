# What Are PC1 and PC2?

## Question

What are PC1 and PC2 doing to help create the 2D images?

## Response

PCA (Principal Component Analysis) finds the directions of greatest variance in a high-dimensional dataset and projects everything onto those directions.

**The problem**: hidden states are 768-dimensional vectors — one number per neuron. You can't plot 768 dimensions directly.

**What PCA does**: looks at all the data points and asks "what single direction through 768-dimensional space captures the most spread/difference between these points?" That direction becomes PC1 (x-axis). Then it asks the same question for the remaining variance, constrained to be perpendicular to PC1 — that becomes PC2 (y-axis).

Like finding the best angle to photograph a 3D sculpture so the most interesting structure is visible in the 2D photo.

**What the percentages mean**: in `word_cluster.py`, PC1 (37.0%) and PC2 (29.5%) together explain 66.5% of total variance. The remaining 33.5% is spread across the other 766 dimensions that got discarded. The 2D plot captures most but not all of the structure.

**What gets lost**: directions in 768D space where points differ but which aren't captured by PC1 or PC2 become invisible. This is why dinner and football have similar PC1/PC2 coordinates in the trajectory plot yet behave very differently — their behavioral difference lives in a low-variance direction that PCA discards.

**Why it still works**: PCA keeps the dimensions where data varies most, so meaningful structure tends to survive. The dinner/treat/walk clustering in `word_cluster.py` is a real signal, not an artifact.
