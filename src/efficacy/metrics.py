"""Evaluation metrics for the efficacy heads.

Deliberately free of any torch dependency so that the metric definitions can be
imported and unit-tested without the model stack, and so that a single
definition is shared by every training and evaluation script rather than being
re-implemented per file (which is how the defects below diverged).

Three defects in releases up to and including v3 are corrected here, and each
one invalidates previously published numbers:

* `enrichment_top5` was scaled by 5 rather than 20. The expected top-5%
  intersection fraction under a random ranking is 0.05, so the random baseline
  for every logged `enrich_top5` value is 0.25, not the 1.0 its comment claimed.
  Applied to the published ASO gene-holdout value of 0.2441, this means top-5%
  selection on unseen target genes performed at ~0.98x random.

* `per_screen_spearman` was computed against the last *feature* tensor instead
  of the label (the label had already been popped off the tuple), so it
  measured feature-vs-prediction correlation. The per-screen median is the
  estimator OligoAI reports, so the head-to-head comparison built on it does
  not hold.

* `pearson_dev` named two different estimators in two different scripts. Use
  `per_item_correlation` and `pooled_correlation` explicitly; never compare one
  against the other.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

TOP_FRACTION = 0.05  # "top 5%"


def top_k_for(n, fraction=TOP_FRACTION):
    return max(1, int(n * fraction))


def enrichment_top5(trues, preds, fraction=TOP_FRACTION):
    """Fold-enrichment of the top-`fraction` predictions, normalised to 1.0 = random.

    E[|top_true & top_pred|] / k = k / n = `fraction` under a random ranking,
    so dividing the observed intersection fraction by `fraction` puts chance at
    1.0 and a perfect ranking at 1 / `fraction` (20.0 for the top 5%).
    """
    trues, preds = np.asarray(trues), np.asarray(preds)
    k = top_k_for(len(trues), fraction)
    top_t = set(np.argsort(-trues)[:k])
    top_p = set(np.argsort(-preds)[:k])
    return float((len(top_t & top_p) / k) / fraction)


def bootstrap_ci(trues, preds, stat="spearman", n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI over paired observations."""
    trues, preds = np.asarray(trues), np.asarray(preds)
    n = len(trues)
    if n < 8:
        return (float("nan"), float("nan"))
    fn = (lambda a, b: spearmanr(a, b).statistic) if stat == "spearman" \
        else (lambda a, b: pearsonr(a, b).statistic)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t, p = trues[idx], preds[idx]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            continue
        v = fn(t, p)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def per_screen_spearman(trues, preds, groups, min_n=8):
    """Median/mean of within-screen Spearman correlations.

    `trues` MUST be the measured labels. Screens with fewer than `min_n` rows,
    or with no variance on either side, are skipped and counted out of
    `n_screens` rather than contributing a degenerate value.
    """
    trues, preds, groups = np.asarray(trues), np.asarray(preds), np.asarray(groups)
    rhos = []
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() < min_n:
            continue
        t, p = trues[sel], preds[sel]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            continue
        v = spearmanr(t, p).statistic
        if np.isfinite(v):
            rhos.append(float(v))
    if not rhos:
        return dict(per_screen_median=None, per_screen_mean=None, n_screens=0)
    return dict(per_screen_median=float(np.median(rhos)),
                per_screen_mean=float(np.mean(rhos)),
                n_screens=len(rhos))


def pooled_correlation(true_mat, pred_mat, mask=None):
    """One Pearson r over the flattened [n_item, n_feature] matrices.

    Also rewards predicting the relative AMPLITUDE of different items, which
    `per_item_correlation` does not. Not interchangeable with it.
    """
    a, b = np.asarray(true_mat), np.asarray(pred_mat)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        a, b = a[m], b[m]
    else:
        a, b = a.ravel(), b.ravel()
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def per_item_correlation(true_mat, pred_mat, mask=None, min_features=10):
    """Mean of within-item Pearson r. Not interchangeable with pooled_correlation."""
    a, b = np.asarray(true_mat), np.asarray(pred_mat)
    m = None if mask is None else np.asarray(mask).astype(bool)
    rs = []
    for i in range(len(a)):
        x, y = (a[i], b[i]) if m is None else (a[i][m[i]], b[i][m[i]])
        if x.size < min_features:
            continue
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            rs.append(float(np.corrcoef(x, y)[0, 1]))
        else:
            rs.append(0.0)
    return float(np.mean(rs)) if rs else float("nan")


def top_k_overlap(true_mat, pred_mat, k=50, mask=None):
    """Mean overlap of the top-k largest-|value| features, per item."""
    a, b = np.asarray(true_mat), np.asarray(pred_mat)
    m = None if mask is None else np.asarray(mask).astype(bool)
    outs = []
    for i in range(len(a)):
        x, y = (a[i], b[i]) if m is None else (a[i][m[i]], b[i][m[i]])
        if x.size < 2:
            continue
        kk = min(k, x.size)
        outs.append(len(set(np.argsort(-np.abs(x))[:kk])
                        & set(np.argsort(-np.abs(y))[:kk])) / kk)
    return float(np.mean(outs)) if outs else float("nan")


def regression_metrics(trues, preds, groups=None, with_ci=False, seed=0):
    """Standard bundle for the efficacy heads."""
    trues, preds = np.asarray(trues), np.asarray(preds)
    out = dict(spearman=float(spearmanr(trues, preds).statistic),
               pearson=float(pearsonr(trues, preds).statistic),
               enrich_top5=enrichment_top5(trues, preds),
               enrich_random_baseline=1.0,
               n=int(len(trues)))
    if groups is not None:
        out.update(per_screen_spearman(trues, preds, groups))
    if with_ci:
        lo, hi = bootstrap_ci(trues, preds, "spearman", seed=seed)
        out["spearman_ci95"] = [lo, hi]
    return out
