# Algorithm

OQBoost 2.0 is a histogram-binned, **2D-oblique** Newton gradient-boosting tree.
Each split is a linear combination of **two** features:

```
coefA·xA + coefB·xB + bias < 0
```

instead of a single axis-aligned threshold. Diagonal and pairwise-interaction
boundaries are represented directly rather than approximated by axis-aligned
staircases.

## Direction finding

The split direction at a node is found by **H-weighted least-squares regression
of the gradient** onto the two features: a one-pass accumulation of 9 scalars and
a 2×2 solve gives the `(coefA, coefB)` that best aligns the residual with the
feature pair. No random projections, no numerical line search. This is the
"gradient-aligned oblique" core — it finds the oblique direction that the
residual actually points along, then thresholds along it.

For each candidate feature pair the projection is histogrammed (`max_bins`) and
the best threshold is chosen by the standard gain criterion. Pairs are searched
in parallel (OpenMP).

## Pair-set breadth — `fast_dir`

`fast_dir` controls *which* feature pairs are evaluated, trading accuracy for
speed (the direction solver above is the same either way):

- `"full"` (default) — every pair, `O(d²)`. Highest accuracy; the right choice
  unless the feature count makes the search too slow.
- `"fast"` — Star anchoring: pair feature 0 with each other feature, `O(d)`. Much
  cheaper on high-dimensional data, slightly lower accuracy.

Legacy integer codes `1` (=full) and `2` (=fast) are still accepted. The earlier
BHC-seed and top-1 modes were removed.

## Multiclass — joint softmax

`multiclass="joint"` (default) trains **one shared 2D-oblique tree per round** with
a per-class leaf-weight vector, on the full softmax gradient
`g_{i,k} = p_k − [y_i=k]`, `h_{i,k} = p_k(1−p_k)`. The 2D direction search needs a
single scalar gradient, so each node reduces the K-class gradient to a
node-consistent **signed** contrast `g_{i,k1} − g_{i,k2}`, where `k1`/`k2` are the
most over- and under-predicted classes by `Σ_i g_{i,k}` (opposite signs → a
discriminative axis); the leaf weights still update all K classes via `−G_k/H_k`.
Each class is initialized to its sample-weighted **log-prior** (mirroring binary's
`logit(prior)`), so imbalanced targets and `class_weight` start from the right
offset instead of a uniform `1/K`. `multiclass="ovr"` is the alternative
(one binary booster per class, probabilities normalized).

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
and small bins keep the pair search cheap. `NaN` is routed to a dedicated learned
bin. Categorical columns marked via `categorical_features` are target-encoded
first (so the oblique split sees a meaningful ordering — [guide](../guides/categorical.md)).

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
split. (This once weakened `categorical_features`, which is now resolved by
target-encoding categoricals instead of relying on 1D isolation — see the
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
