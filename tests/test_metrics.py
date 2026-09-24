"""Regression tests for the evaluation-metric defects fixed in v4.

These are deliberately dependency-light (numpy + scipy only, no torch) so that
they run in CI and on a laptop without the ~3 GB of model/data assets. They pin
down the three metric defects that invalidated previously published numbers:

  1. top-5% enrichment was scaled by 5, making 0.25 (not 1.0) the random
     baseline -- a 4x understatement that also made the quoted "5.72x random"
     target bar dimensionally incomparable.
  2. per-screen Spearman was computed against the last FEATURE tensor instead
     of the label, so it measured feature-vs-prediction correlation.
  3. "pearson_dev" named two different estimators in two different scripts
     (mean of within-perturbation r, vs one r over the flattened matrix), and
     the published improvement subtracted one from the other.

Run with:  python -m pytest tests/ -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "efficacy"))


# --------------------------------------------------------------------------
# 1. enrichment normalisation
# --------------------------------------------------------------------------

def _enrich(trues, preds, factor):
    k = max(1, len(trues) // 20)
    top_t = set(np.argsort(-trues)[:k])
    top_p = set(np.argsort(-preds)[:k])
    return len(top_t & top_p) / k * factor


def test_random_ranking_gives_enrichment_of_one():
    """Under a random ranking the normalised enrichment must centre on 1.0."""
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(300):
        y = rng.normal(size=2000)
        p = rng.normal(size=2000)          # independent of y
        vals.append(_enrich(y, p, 20.0))
    assert 0.9 < float(np.mean(vals)) < 1.1, (
        f"normalised enrichment under random ranking should be ~1.0, "
        f"got {np.mean(vals):.3f}")


def test_legacy_factor_five_understates_by_four():
    """The pre-v4 factor of 5 puts the random baseline at 0.25, not 1.0."""
    rng = np.random.default_rng(1)
    vals = [_enrich(rng.normal(size=2000), rng.normal(size=2000), 5.0)
            for _ in range(300)]
    assert 0.2 < float(np.mean(vals)) < 0.3
    # Hence the published gene-holdout value of 0.2441 is ~0.98x random.
    assert 0.9 < 0.2441 / 0.25 < 1.06


def test_perfect_ranking_saturates():
    y = np.arange(2000, dtype=float)
    assert _enrich(y, y.copy(), 20.0) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# 2. per-screen Spearman must use the label
# --------------------------------------------------------------------------

def test_per_screen_uses_labels_not_features():
    """Reproduces the v2/v3 defect: popping the label then indexing [-1].

    The tuple returned by prep() is (...features..., label). After `sb = sb[:-1]`
    removes the label, `sb[-1]` is the last feature. Correlating that with the
    predictions is unrelated to accuracy -- here the model is deliberately made
    to track the feature while being anti-correlated with the truth.
    """
    from metrics import per_screen_spearman

    rng = np.random.default_rng(2)
    n = 400
    feature = rng.normal(size=n)
    y_true = rng.normal(size=n)
    preds = feature * 3.0 - y_true * 0.05     # follows the feature, not the label
    groups = np.repeat(np.arange(20), n // 20)

    correct = per_screen_spearman(y_true, preds, groups)["per_screen_median"]
    buggy = per_screen_spearman(feature, preds, groups)["per_screen_median"]

    assert buggy > 0.9, "the buggy estimator should look near-perfect"
    assert abs(correct) < 0.4, "the correct estimator should show the model is poor"
    assert buggy - correct > 0.5


def test_per_screen_skips_degenerate_screens():
    from metrics import per_screen_spearman
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 1, 1, 1, 1])
    p = np.arange(16, dtype=float)
    g = np.array(["a"] * 8 + ["b"] * 8)       # screen "b" has zero label variance
    out = per_screen_spearman(y, p, g, min_n=8)
    assert out["n_screens"] == 1


def test_per_screen_respects_min_n():
    from metrics import per_screen_spearman
    y = np.arange(10, dtype=float)
    p = y + 0.1
    g = np.array(["a"] * 5 + ["b"] * 5)
    assert per_screen_spearman(y, p, g, min_n=8)["n_screens"] == 0


# --------------------------------------------------------------------------
# 3. the two "pearson_dev" estimators are not interchangeable
# --------------------------------------------------------------------------

def _per_pert(true, pred):
    return float(np.mean([np.corrcoef(true[i], pred[i])[0, 1]
                          for i in range(len(true))]))


def _pooled(true, pred):
    return float(np.corrcoef(true.ravel(), pred.ravel())[0, 1])


def test_pooled_exceeds_per_pert_when_between_item_levels_differ():
    """Pooling additionally rewards getting each item's overall LEVEL right.

    Per-item correlation centres and scales within each perturbation, so a
    correctly predicted between-perturbation offset contributes nothing to it.
    The pooled estimator does not centre per item, so that same offset drives
    the correlation on its own.

    Here every within-profile deviation is pure noise -- the model has no
    gene-level signal whatsoever -- yet pooled reads high. Quoting a pooled
    number against a per-item number, as the published 0.31 -> 0.71 comparison
    does, can therefore manufacture an improvement out of a change of
    estimator alone.
    """
    rng = np.random.default_rng(3)
    n_pert, n_gene = 60, 400
    level = rng.normal(scale=5.0, size=n_pert)          # per-perturbation offset
    true = level[:, None] + rng.normal(size=(n_pert, n_gene))
    pred = level[:, None] + rng.normal(size=(n_pert, n_gene))

    pp, po = _per_pert(true, pred), _pooled(true, pred)
    assert abs(pp) < 0.1, f"within-perturbation signal should be ~0, got {pp:.3f}"
    assert po > 0.7, f"pooled should be inflated by level alone, got {po:.3f}"


def test_estimators_agree_when_amplitudes_are_equal():
    """With a common scale the two estimators are close, which is why the
    discrepancy went unnoticed until datasets with different panels were mixed."""
    rng = np.random.default_rng(4)
    true = rng.normal(size=(60, 400))
    pred = 0.5 * true + rng.normal(size=(60, 400))
    assert abs(_per_pert(true, pred) - _pooled(true, pred)) < 0.05


# --------------------------------------------------------------------------
# 4. masking unmeasured columns
# --------------------------------------------------------------------------

def test_unmeasured_columns_inflate_unmasked_correlation():
    """Columns a dataset never measured carry dev == -common_fc for EVERY
    perturbation. Scoring them rewards reproducing a dataset-level constant.
    """
    rng = np.random.default_rng(5)
    n_pert, n_meas, n_unmeas = 40, 200, 800
    common = rng.normal(scale=2.0, size=n_unmeas)      # the constant column block

    true_meas = rng.normal(size=(n_pert, n_meas))
    pred_meas = rng.normal(size=(n_pert, n_meas))      # no real signal
    true_un = np.tile(-common, (n_pert, 1))
    pred_un = np.tile(-common, (n_pert, 1))            # trivially predictable

    true = np.concatenate([true_meas, true_un], axis=1)
    pred = np.concatenate([pred_meas, pred_un], axis=1)
    mask = np.concatenate([np.ones((n_pert, n_meas), bool),
                           np.zeros((n_pert, n_unmeas), bool)], axis=1)

    unmasked = _per_pert(true, pred)
    masked = float(np.mean([np.corrcoef(true[i][mask[i]], pred[i][mask[i]])[0, 1]
                            for i in range(n_pert)]))
    assert unmasked > 0.8, f"unmasked score should be inflated, got {unmasked:.3f}"
    assert abs(masked) < 0.1, f"masked score should reveal no signal, got {masked:.3f}"


# --------------------------------------------------------------------------
# 5. bootstrap CI sanity
# --------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    from metrics import bootstrap_ci
    from scipy.stats import spearmanr
    rng = np.random.default_rng(6)
    y = rng.normal(size=500)
    p = 0.6 * y + rng.normal(size=500)
    lo, hi = bootstrap_ci(y, p, n_boot=400, seed=0)
    rho = spearmanr(y, p).statistic
    assert lo < rho < hi
    assert hi - lo < 0.4


def test_bootstrap_ci_is_wider_for_small_n():
    from metrics import bootstrap_ci
    rng = np.random.default_rng(7)
    y_big = rng.normal(size=800)
    p_big = 0.5 * y_big + rng.normal(size=800)
    lo_b, hi_b = bootstrap_ci(y_big, p_big, n_boot=400, seed=0)
    lo_s, hi_s = bootstrap_ci(y_big[:40], p_big[:40], n_boot=400, seed=0)
    assert (hi_s - lo_s) > (hi_b - lo_b)
