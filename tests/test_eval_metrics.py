"""Tests for the shared matrix-level metrics used by the response head.

Mirrors tests/test_metrics.py for the efficacy line. The point of a single
shared module is that the two `pearson_dev` estimators cannot silently diverge
again (docs/ERRATA.md, E4), so the properties that distinguish them are pinned
here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "maprna_p3"))

from eval_metrics import (  # noqa: E402
    bootstrap_ci_per_item, masked_mse, per_item_correlation, pooled_correlation,
    top_k_overlap)


def test_identical_matrices_score_one():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(20, 100))
    assert per_item_correlation(m, m) == pytest.approx(1.0)
    assert pooled_correlation(m, m) == pytest.approx(1.0)
    assert top_k_overlap(m, m, k=10) == pytest.approx(1.0)


def test_estimators_diverge_on_between_item_level():
    """The defining difference: pooled sees item level, per-item does not."""
    rng = np.random.default_rng(1)
    level = rng.normal(scale=5.0, size=40)
    true = level[:, None] + rng.normal(size=(40, 200))
    pred = level[:, None] + rng.normal(size=(40, 200))
    assert abs(per_item_correlation(true, pred)) < 0.1
    assert pooled_correlation(true, pred) > 0.7


def test_mask_excludes_constant_columns():
    """A block of per-dataset-constant columns must not create signal."""
    rng = np.random.default_rng(2)
    n, meas, unmeas = 30, 100, 400
    const = rng.normal(scale=2.0, size=unmeas)
    true = np.concatenate([rng.normal(size=(n, meas)), np.tile(-const, (n, 1))], axis=1)
    pred = np.concatenate([rng.normal(size=(n, meas)), np.tile(-const, (n, 1))], axis=1)
    mask = np.concatenate([np.ones((n, meas), bool), np.zeros((n, unmeas), bool)], axis=1)
    assert per_item_correlation(true, pred) > 0.8          # unmasked: inflated
    assert abs(per_item_correlation(true, pred, mask)) < 0.1


def test_masked_mse_normalises_per_item():
    true = np.zeros((3, 10))
    pred = np.ones((3, 10))
    mask = np.zeros((3, 10), bool)
    mask[:, :4] = True                                      # 4 measured columns
    assert masked_mse(true, pred, mask) == pytest.approx(1.0)
    assert masked_mse(true, pred) == pytest.approx(1.0)


def test_items_with_too_few_measured_features_are_skipped():
    rng = np.random.default_rng(3)
    true = rng.normal(size=(4, 50))
    pred = true.copy()
    mask = np.ones((4, 50), bool)
    mask[0, :] = False
    mask[0, :3] = True                                      # only 3 measured
    # the degenerate item is dropped rather than contributing a noisy r
    assert per_item_correlation(true, pred, mask, min_features=10) == pytest.approx(1.0)


def test_degenerate_input_returns_nan_not_an_exception():
    flat = np.ones((5, 20))
    rng = np.random.default_rng(4)
    assert np.isnan(pooled_correlation(flat, rng.normal(size=(5, 20))))


def test_bootstrap_resamples_items_and_brackets_the_estimate():
    rng = np.random.default_rng(5)
    true = rng.normal(size=(60, 120))
    pred = 0.7 * true + rng.normal(size=(60, 120))
    point = per_item_correlation(true, pred)
    lo, hi = bootstrap_ci_per_item(true, pred, n_boot=300, seed=0)
    assert lo < point < hi


def test_top_k_overlap_is_zero_for_disjoint_extremes():
    true = np.zeros((1, 100))
    pred = np.zeros((1, 100))
    true[0, :10] = 10.0                                     # extremes at the front
    pred[0, -10:] = 10.0                                    # extremes at the back
    assert top_k_overlap(true, pred, k=10) == 0.0
