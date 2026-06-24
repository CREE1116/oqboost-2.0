# Categorical features

Pass `categorical_features` (column indices or a boolean mask) to give those
columns **lossless binning** — one histogram bin per level, ignoring `max_bins`.

```python
clf = OQBoostClassifier(categorical_features=[0, 4]).fit(X, y)
# or a boolean mask
clf = OQBoostClassifier(categorical_features=[True, False, False, False, True]).fit(X, y)
```

Encode categories as **integer codes** (`0..K-1`) before fitting (e.g. with
`sklearn.preprocessing.OrdinalEncoder`).

## Why a flag is needed

Without the flag, integer codes are binned like continuous values via quantiles.
When cardinality exceeds `max_bins`, distinct levels get merged into the same bin
— a lossless-to-lossy collapse that destroys high-cardinality categorical signal.
Marking the column forces one bin per level, so no merging occurs while continuous
features keep their low-resolution (direction-stable) binning.

## Known limitation (status: under revision)

> The lossless-binning benefit has **weakened** since 1D-split competition was
> removed from the core. Isolating a single category level cleanly wants a 1D
> axis threshold (a cut between two adjacent codes); with the 2D-only search a
> category is paired with a continuous feature, which dilutes the isolation.

Restoring 1D competition is not the intended fix (it complicates the core). A
dedicated categorical-encoding strategy (out-of-fold / ordered target statistics,
or an oblique-aware encoding) is the planned direction — see the
[roadmap](../internals/roadmap.md). For now, `categorical_features` still helps on
genuinely high-cardinality columns but less than it once did; on low-cardinality
nominal data the effect is small either way.
