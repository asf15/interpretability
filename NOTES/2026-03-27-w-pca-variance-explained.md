# PCA Variance Percentage — What It Really Means

## Question

So a high % variance means most of the differences will be visible, but a low % variance means the differences are small?

## Response

Almost — but there's a subtle distinction:

**High % variance** means the 2D projection is capturing most of the *spread* in the dataset, so the relative positions of points are reliable. Things that look close together actually are close, and things that look far apart actually are far.

**Low % variance** doesn't mean the differences are small — it means the differences are *hiding in dimensions PCA didn't choose*. The dinner vs football behavioral difference is a good example: their PC1/PC2 coordinates are close together, but they produce very different outputs. The difference is real and large — it's just concentrated in dimensions 3-768 rather than dimensions 1-2.

Low % variance captured is about the *reliability of the picture*, not the *size of the differences*. A 2D plot capturing only 30% of variance might miss important structure entirely, while a plot capturing 90% is showing almost everything.

## Applied to the Visualizations

- **Trajectory plot**: 67.0% + 26.9% = 93.9% — seeing almost all the structure, picture is very reliable. But dinner vs football difference was hidden in the remaining 6.1%.
- **Word cluster plot**: 37.0% + 29.5% = 66.5% — reasonably confident the clustering is real, but some structure is missing.
