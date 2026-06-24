# LOB — Lineage Oblique Boosting (experimental)

LOB is an opt-in extension (`max_lineage > 0`, default `0` = classic 2D) that
**approximates high-order oblique interactions using only 2×2 solves**.

```python
clf = OQBoostClassifier(max_lineage=2, n_screen=16).fit(X, y)
```

## Idea

A classic 2D node pairs two raw features. A LOB node additionally inherits the
**directions** of its ancestors: the projection `z = coefA·xA + coefB·xB` computed
at a parent becomes a candidate "feature" at the child. The child then searches
`(z, raw)` and `(z, z)` pairs, so directions **compose hierarchically** down the
tree — a depth-`k` path can express a `k`-fold oblique combination without ever
doing a full d-dimensional solve, only stacked 2×2 fits.

Inherited directions are stored in a dense `dirs_` table; each node carries a
`dir_id`. `max_lineage` bounds how many ancestor directions a node may reuse.

## Screening

The candidate pool grows with inherited directions, so screening matters: the
root is searched exhaustively, deeper nodes use SIS (sure independence screening)
to keep the top-`n_screen` candidates. Pair `max_lineage` with `n_screen` (e.g.
16) to bound cost.

## When it helps

LOB is **oblique-only** — axis trees cannot inherit directions. It gains on data
with genuine high-order oblique + interaction structure (synthetic composed XOR:
+0.02 AUC). On typical real data the gain is marginal (~+0.0016 mean) because the
boosting ensemble already fills much of that gap by composition. Treat it as
experimental.

## Limitations

- `explain()` is **not supported** with `max_lineage > 0` — composed dense
  directions have no path-additive attribution.
- Higher memory (the `dirs_` table) and a larger candidate search.
