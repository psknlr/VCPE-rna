"""Matrix-level evaluation metrics for the perturbation-response head.

Torch-free, so the definitions can be unit-tested without the model stack, and
shared between `train_p3.py`, `eval_fair.py` and the baselines rather than
re-implemented per script -- per-file re-implementation is how the two
`pearson_dev` estimators came to diverge (see docs/ERRATA.md, E4).

Two rules encoded here:

1. **Name the estimator.** `per_item_correlation` (mean of within-perturbation
   r) and `pooled_correlation` (one r over the flattened matrix) answer
   different questions. The pooled form additionally rewards predicting the
   relative LEVEL of different perturbations, so on the same checkpoint the two
   can differ by ~1.5x. They must never be subtracted from one another.

2. **Mask unmeasured entries.** An HVG column that a dataset's panel never
   measured has `fc == 0` by construction, hence `dev == -common_fc` for every
   perturbation of that dataset -- a per-dataset constant with no
   perturbation-specific content. Scoring it rewards reproducing that constant
   (docs/ERRATA.md, E6).
"""
import numpy as np


def _rows(true_mat, pred_mat, mask):
    a, b = np.asarray(true_mat), np.asarray(pred_mat)
    m = None if mask is None else np.asarray(mask).astype(bool)
    for i in range(len(a)):
        if m is None:
            yield a[i], b[i]
        else:
            yield a[i][m[i]], b[i][m[i]]


def pooled_correlation(true_mat, pred_mat, mask=None):
    """One Pearson r over all (masked) entries of the flattened matrices."""
    a, b = np.asarray(true_mat), np.asarray(pred_mat)
    if mask is None:
        x, y = a.ravel(), b.ravel()
    else:
        m = np.asarray(mask).astype(bool)
        x, y = a[m], b[m]
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def per_item_correlation(true_mat, pred_mat, mask=None, min_features=10):
    """Mean of within-item Pearson r over (masked) features."""
    rs = []
    for x, y in _rows(true_mat, pred_mat, mask):
        if x.size < min_features:
            continue
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            rs.append(float(np.corrcoef(x, y)[0, 1]))
        else:
            rs.append(0.0)
    return float(np.mean(rs)) if rs else float("nan")


def top_k_overlap(true_mat, pred_mat, k=50, mask=None):
    """Mean overlap of the top-k largest-|value| features, per item."""
    outs = []
    for x, y in _rows(true_mat, pred_mat, mask):
        if x.size < 2:
            continue
        kk = min(k, x.size)
        outs.append(len(set(np.argsort(-np.abs(x))[:kk])
                        & set(np.argsort(-np.abs(y))[:kk])) / kk)
    return float(np.mean(outs)) if outs else float("nan")


def masked_mse(true_mat, pred_mat, mask=None):
    """Mean over items of the within-item mean squared error on measured entries."""
    a, b = np.asarray(true_mat), np.asarray(pred_mat)
    if mask is None:
        return float(((a - b) ** 2).mean())
    m = np.asarray(mask).astype(float)
    d2 = ((a - b) ** 2) * m
    denom = np.maximum(m.sum(axis=1), 1.0)
    return float((d2.sum(axis=1) / denom).mean())


def bootstrap_ci_per_item(true_mat, pred_mat, mask=None, n_boot=1000,
                          alpha=0.05, seed=0):
    """Percentile CI for `per_item_correlation`, resampling ITEMS.

    Items (perturbations) are the independent unit here; resampling individual
    gene entries would treat genes within a profile as independent and give a
    misleadingly tight interval.
    """
    a = np.asarray(true_mat)
    n = len(a)
    if n < 8:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = None if mask is None else np.asarray(mask)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = per_item_correlation(a[idx], np.asarray(pred_mat)[idx],
                                 None if m is None else m[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
