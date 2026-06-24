# Algorithm

OQBoost 2.0 is a histogram-binned, **2D-oblique** Newton gradient-boosting tree.
Each split is a linear combination of **two** features:

```
coefA·xA + coefB·xB + bias < 0
```

instead of a single axis-aligned threshold. Diagonal and pairwise-interaction
boundaries are represented directly rather than approximated by axis-aligned
staircases.

## Direction finding — `fast_dir`

The split direction at a node is found by **H-weighted least-squares regression
of the gradient** onto the two features: a one-pass accumulation of 9 scalars and
a 2×2 solve gives the `(coefA, coefB)` that best aligns the residual with the
feature pair. No random projections, no numerical line search. This is the
"gradient-aligned oblique" core — it finds the oblique direction that the
residual actually points along, then thresholds along it.

For each candidate feature pair the projection is histogrammed (`max_bins`) and
the best threshold is chosen by the standard gain criterion. Pairs are searched
in parallel (OpenMP).

## Newton boosting

Gradient and hessian per sample drive leaf values (`-G/H`), the split gain, and
the direction fit:

- Classification (logistic): `g = p − y`, `h = p(1−p)`.
- Regression (squared): `g = raw − y`, `h = 1`.

Robust regression losses (`huber`, `quantile`) clip or sign the gradient and, for
`quantile`, recompute leaf values as the target quantile of residuals.

## Histogram binning

Features are globally pre-binned into `max_bins` quantile bins once. Keep
`max_bins` small (default 16) — the oblique direction is stable at low resolution,
and small bins keep the pair search cheap. `categorical_features` overrides this
with lossless per-level binning ([guide](../guides/categorical.md)). `NaN` is
routed to a dedicated learned bin.

## Higher-order interactions

A single split is 2D, but **depth and boosting stack 2D atoms** into higher-order
structure: a depth-`k` path composes `k` oblique cuts, and the additive ensemble
sums many such trees. There is no full d-dimensional oblique solve — the
expressiveness comes from composition. (The experimental
[LOB](lob.md) extension approximates higher-order oblique interactions with only
2×2 solves by inheriting ancestor directions.)

## 1D competition removed

Earlier versions let a 1D axis split compete with the 2D oblique split per node.
Measurement showed 2D subsumes 1D (a 2D split with `b ≈ 0` *is* a 1D split): mean
ΔAUC of removing the competition was −0.0003 across the binary suite. The 1D path
is kept only as a fallback for degenerate nodes where the 2D search finds no
split. (One regression from this: it weakened `categorical_features` — see the
[categorical guide](../guides/categorical.md).)

## Threading

Pair search parallelizes over feature pairs; per-sample loops (gradient, raw-score
accumulation, prediction) parallelize over samples. Nodes below a work threshold
fall back to serial to avoid fork-join overhead, so small-data fits don't regress
with more threads. Results are bit-identical regardless of thread count. Batch
inference uses a tree-outer layout (per-thread sample chunk, tree loop inside) for
cache-hot nodes — ~1.4× faster, bit-identical.

## Performance ceiling

Accuracy sits at the GBDT-paradigm ceiling. Exotic extensions (LOB, linear leaves,
direction diversity, ordered boosting/encoding) all plateau at ~0.01 marginal —
the additive residual loop already subsumes the extra expressiveness and supplies
its own regularization. See the [roadmap](roadmap.md).
