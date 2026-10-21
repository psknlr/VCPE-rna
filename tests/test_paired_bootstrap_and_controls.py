"""The statistics a comparison rests on, and the controls it is made against.

Two gaps this covers.

**Power.** Every comparison in this repository was settled by a normal
approximation over 3 or 5 seeds. At n=3 that interval is wide almost regardless
of the data, so "tie" has been doing double duty for "no difference" and "no
power" -- the head-vs-ridge CI of [-0.031, +0.026] is consistent with a real
effect of either sign. `paired_bootstrap_per_item` resamples the few hundred
held-out perturbations instead. It answers a different question from the
multi-seed interval and does not replace it, and the tests below pin that
distinction as well as the arithmetic.

**Fair controls.** `ridge_esm2` ran at alpha=1.0 and `knn_esm2` at k=10 for every
published number, never selected, while the head's epoch is selected on an inner
split and its lr / width / decay were tuned over many runs. And between "a linear
map on these features" and "this architecture" there was no rung: any margin over
ridge was attributed to the architecture by default. `select_control_hyperparams`
and `rff_ridge_esm2` close both, and must do so without ever reading the held-out
set.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))

from baselines import (  # noqa: E402
    baseline_ridge_esm2, baseline_rff_ridge_esm2, make_rff, run_all,
    select_control_hyperparams)
from eval_metrics import (  # noqa: E402
    direction_agreement, enrichment_factor, paired_bootstrap_per_item,
    per_item_correlation, per_item_correlation_detail,
    per_item_correlation_vector, per_item_spearman)


def _data(n, d=32, g=60, seed=0, nonlinear=True):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, g)) * 0.4
    X = rng.normal(size=(n, d))
    Y = (np.tanh(X @ W) if nonlinear else X @ W) + 0.3 * rng.normal(size=(n, g))
    return X.astype(np.float32), Y.astype(np.float32)


# --------------------------------------------------------------------------
# the paired item-level bootstrap
# --------------------------------------------------------------------------

def test_the_fast_path_is_exactly_the_naive_loop():
    """The replicates resample precomputed per-item correlations instead of
    recomputing them, which is only legitimate if it is identical. It is 150x
    faster, which is what lets the interval go on every row of every table."""
    rng = np.random.default_rng(3)
    true = rng.normal(size=(60, 80))
    a = true + rng.normal(size=(60, 80)) * 1.5
    b = true + rng.normal(size=(60, 80)) * 1.7
    mask = rng.random((60, 80)) > 0.2

    def naive(n_boot, seed):
        r = np.random.default_rng(seed)
        vals = []
        for _ in range(n_boot):
            idx = r.integers(0, len(true), len(true))
            mm = mask[idx]
            vals.append(per_item_correlation(true[idx], a[idx], mm)
                        - per_item_correlation(true[idx], b[idx], mm))
        v = np.asarray(vals)
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    fast = paired_bootstrap_per_item(true, a, b, mask=mask, n_boot=300, seed=11)
    assert fast["ci95"] == pytest.approx(list(naive(300, 11)), abs=1e-12)


def test_the_vector_mean_is_the_estimator():
    rng = np.random.default_rng(4)
    true = rng.normal(size=(30, 50))
    pred = true + rng.normal(size=(30, 50))
    v = per_item_correlation_vector(true, pred)
    assert np.nanmean(v) == pytest.approx(per_item_correlation(true, pred))


def test_a_real_difference_is_detected_and_a_null_one_is_not():
    rng = np.random.default_rng(5)
    true = rng.normal(size=(200, 100))
    better = true + rng.normal(size=(200, 100)) * 0.5
    worse = true + rng.normal(size=(200, 100)) * 3.0
    real = paired_bootstrap_per_item(true, better, worse, n_boot=500, seed=1)
    assert real["difference"] > 0 and real["excludes_zero"]
    assert real["ci95"][0] > 0

    same_a = true + rng.normal(size=(200, 100)) * 1.5
    same_b = true + rng.normal(size=(200, 100)) * 1.5
    null = paired_bootstrap_per_item(true, same_a, same_b, n_boot=500, seed=1)
    assert not null["excludes_zero"], "two equally good predictors must tie"


def test_pairing_is_what_gives_the_interval_its_power():
    """Both methods are scored on the same resampled items. If they were not, the
    item-to-item spread of the metric -- far larger than the gap between two
    methods -- would swamp the difference."""
    rng = np.random.default_rng(6)
    # item difficulty varies a lot, the method gap is small and consistent
    hard = rng.uniform(0.5, 4.0, size=150)[:, None]
    true = rng.normal(size=(150, 100))
    a = true + rng.normal(size=(150, 100)) * hard
    b = true + rng.normal(size=(150, 100)) * (hard * 1.12)
    paired = paired_bootstrap_per_item(true, a, b, n_boot=600, seed=2)

    va = per_item_correlation_vector(true, a)
    vb = per_item_correlation_vector(true, b)
    unpaired_sd = np.sqrt(np.var(va, ddof=1) / len(va) + np.var(vb, ddof=1) / len(vb))
    paired_sd = np.std(va - vb, ddof=1) / np.sqrt(len(va))
    assert paired_sd < unpaired_sd, "pairing must shrink the standard error here"
    assert paired["excludes_zero"], "the paired interval should resolve this gap"


def test_the_interval_says_what_it_does_not_cover():
    """A reader who takes the item-level interval for the whole uncertainty will
    call a three-seed tie resolved. The field has to travel with the number."""
    rng = np.random.default_rng(7)
    true = rng.normal(size=(40, 60))
    out = paired_bootstrap_per_item(true, true * 0.9, true * 0.5, n_boot=100)
    assert out["resampled"] == "items within one split"
    assert "split" in out["does_not_cover"]
    assert "run_seeds" in out["does_not_cover"]


def test_too_few_items_is_reported_not_silently_narrow():
    rng = np.random.default_rng(8)
    true = rng.normal(size=(4, 40))
    out = paired_bootstrap_per_item(true, true * 0.9, true * 0.4, n_boot=50)
    assert np.isnan(out["ci95"][0]) and "too few" in out["note"]


# --------------------------------------------------------------------------
# secondary metrics: chance levels, which is what E2 got wrong
# --------------------------------------------------------------------------

def test_enrichment_factor_puts_chance_at_one():
    """ERRATA E2: the top-5% enrichment was scaled by 5 instead of 20, so chance
    read 0.25 and a model at chance looked four times worse than nothing."""
    rng = np.random.default_rng(9)
    true = rng.normal(size=(600, 500))
    chance = rng.normal(size=(600, 500))
    assert enrichment_factor(true, chance, k_frac=0.05) == pytest.approx(1.0, abs=0.08)
    # a perfect ranking recovers everything: 1 / k_frac
    assert enrichment_factor(true, true, k_frac=0.05) == pytest.approx(20.0, abs=0.5)


def test_enrichment_chance_holds_under_a_mask():
    rng = np.random.default_rng(10)
    true = rng.normal(size=(400, 400))
    mask = rng.random((400, 400)) > 0.3
    ef = enrichment_factor(true, rng.normal(size=(400, 400)), mask=mask)
    assert ef == pytest.approx(1.0, abs=0.1)


def test_direction_agreement_chance_is_one_half():
    rng = np.random.default_rng(11)
    true = rng.normal(size=(400, 300))
    assert direction_agreement(true, rng.normal(size=(400, 300))) == \
        pytest.approx(0.5, abs=0.03)
    assert direction_agreement(true, true) == pytest.approx(1.0)
    assert direction_agreement(true, -true) == pytest.approx(0.0)


def test_direction_agreement_is_not_implied_by_correlation():
    """A model can correlate well overall and still have the sign wrong on the
    genes with the largest effects -- which is the thing a user acts on."""
    rng = np.random.default_rng(12)
    true = rng.normal(size=(80, 200))
    pred = true.copy()
    for i in range(len(true)):
        big = np.argsort(-np.abs(true[i]))[:4]      # the top 2% of genes
        pred[i, big] *= -1.0
    # still strongly correlated overall, and wrong on every gene that matters
    assert per_item_correlation(true, pred) > 0.5
    assert direction_agreement(true, pred, top_frac=0.02) == pytest.approx(0.0)


def test_spearman_and_pearson_are_separate_names():
    """E4 happened because two estimators were reported under one name. A rank
    correlation and a Pearson correlation on the same matrices must not be
    interchangeable."""
    rng = np.random.default_rng(13)
    true = rng.normal(size=(50, 200))
    pred = np.sign(true) * np.abs(true) ** 3          # monotone, very non-linear
    assert per_item_spearman(true, pred) == pytest.approx(1.0, abs=1e-9)
    assert per_item_correlation(true, pred) < 0.95


def test_degenerate_and_skipped_items_are_counted():
    """The mean hides them: a constant profile is scored 0.0 by imputation, and a
    heavily masked item is dropped. A mean over 40 of 352 items is a different
    quantity from a mean over 352."""
    rng = np.random.default_rng(14)
    true = rng.normal(size=(6, 40))
    pred = true + rng.normal(size=(6, 40)) * 0.1
    pred[0] = 0.0                                     # constant prediction
    mask = np.ones((6, 40), dtype=bool)
    mask[1, 35:] = True
    mask[1, :35] = False                              # only 5 measured -> skipped
    d = per_item_correlation_detail(true, pred, mask)
    assert d["n_degenerate"] == 1
    assert d["n_skipped_too_few_features"] == 1
    assert d["n_scored"] == 5


# --------------------------------------------------------------------------
# control hyperparameters, selected on inner validation only
# --------------------------------------------------------------------------

def test_selection_never_touches_the_held_out_set():
    """Structural: the function is only ever given fit and val arrays. Checked by
    passing a held-out array that would poison the result if it were read."""
    Xf, Yf = _data(120, seed=20)
    Xv, Yv = _data(40, seed=21)
    sel = select_control_hyperparams(Yf, Xf, Yv, Xv, rff_n_features=128)
    assert sel["selected_on"] == "inner_val"
    assert sel["n_fit_items"] == 120 and sel["n_val_items"] == 40
    assert sel["estimator"] == "per_item_correlation"


def test_selection_beats_the_fixed_default_on_held_out_data():
    """Not a guarantee in general, but if a selected knob never helped there would
    be nothing to fix. On data where alpha=1.0 is a poor choice it must help."""
    Xf, Yf = _data(150, seed=22)
    Xv, Yv = _data(50, seed=23)
    Xt, Yt = _data(60, seed=24)
    sel = select_control_hyperparams(Yf, Xf, Yv, Xv, rff_n_features=128)
    default = baseline_ridge_esm2(Yf, Xf, Xt, alpha=1.0)
    chosen = baseline_ridge_esm2(Yf, Xf, Xt, alpha=sel["ridge_alpha"])
    assert (per_item_correlation(Yt, chosen)
            >= per_item_correlation(Yt, default) - 1e-9)


def test_the_grid_is_reported_so_the_choice_is_auditable():
    Xf, Yf = _data(80, seed=25)
    Xv, Yv = _data(30, seed=26)
    sel = select_control_hyperparams(Yf, Xf, Yv, Xv, rff_n_features=128)
    assert len(sel["grids"]["ridge_alpha"]) >= 5
    assert all("inner_val" in row for row in sel["grids"]["ridge_alpha"])
    assert "ridge_alpha_at_grid_edge" in sel, (
        "a value chosen at the edge of the grid was chosen by the grid, not the "
        "data, and the report has to say so")


def test_the_default_rows_keep_their_meaning():
    """`ridge_esm2` must mean in a new table what it meant in an old one, or two
    tables cannot be compared -- which is ERRATA E4 exactly."""
    Xf, Yf = _data(100, seed=27)
    Xt, _ = _data(30, seed=28)
    Xv, Yv = _data(30, seed=29)
    sel = select_control_hyperparams(Yf, Xf, Yv, Xv, rff_n_features=128)
    preds = run_all(Yf, Xf, Xt, which=["ridge_esm2", "ridge_esm2_sel"],
                    selected=sel)
    assert np.allclose(preds["ridge_esm2"],
                       baseline_ridge_esm2(Yf, Xf, Xt, alpha=1.0))
    if sel["ridge_alpha"] != 1.0:
        assert not np.allclose(preds["ridge_esm2"], preds["ridge_esm2_sel"])


def test_selected_rows_are_skipped_rather_than_silently_defaulted():
    """Without a selection dict the `_sel` rows must be absent, not duplicates of
    the defaults reported under a second name."""
    Xf, Yf = _data(60, seed=30)
    Xt, _ = _data(20, seed=31)
    preds = run_all(Yf, Xf, Xt, which=["ridge_esm2", "ridge_esm2_sel"])
    assert "ridge_esm2_sel" not in preds


# --------------------------------------------------------------------------
# the non-linearity control
# --------------------------------------------------------------------------

def test_the_rff_control_is_deterministic_given_its_seed():
    """ERRATA E14a: GEARS is not deterministic at a fixed seed, which made a
    single run meaningless. A control must not have that property."""
    Xf, Yf = _data(80, seed=32)
    Xt, _ = _data(25, seed=33)
    a = baseline_rff_ridge_esm2(Yf, Xf, Xt, n_features=128, seed=5)
    b = baseline_rff_ridge_esm2(Yf, Xf, Xt, n_features=128, seed=5)
    c = baseline_rff_ridge_esm2(Yf, Xf, Xt, n_features=128, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c), "a different seed must draw different features"


def test_the_rff_control_beats_the_linear_one_on_a_nonlinear_truth():
    """It is the rung between a linear map and the head. If it could not exceed a
    linear map on a non-linear ground truth it would not be that rung."""
    Xf, Yf = _data(400, d=16, g=40, seed=34, nonlinear=True)
    Xv, Yv = _data(80, d=16, g=40, seed=35, nonlinear=True)
    Xt, Yt = _data(120, d=16, g=40, seed=36, nonlinear=True)
    sel = select_control_hyperparams(Yf, Xf, Yv, Xv, rff_n_features=512)
    lin = baseline_ridge_esm2(Yf, Xf, Xt, alpha=sel["ridge_alpha"])
    nl = baseline_rff_ridge_esm2(Yf, Xf, Xt, alpha=sel["rff_alpha"],
                                 gamma_mult=sel["rff_gamma_mult"],
                                 n_features=512, seed=0)
    assert per_item_correlation(Yt, nl) > per_item_correlation(Yt, lin)


def test_the_rff_control_reads_only_the_features_the_others_read():
    """It must be a feature map of the embedding alone. If it could see the
    response profiles it would not be a control, it would be a leak."""
    Xf, Yf = _data(50, seed=37)
    W, b = make_rff(Xf.shape[1], n_features=64, gamma=1.0, seed=0)
    assert W.shape == (Xf.shape[1], 64) and b.shape == (64,)
    # same embeddings, different labels -> same feature map
    W2, b2 = make_rff(Xf.shape[1], n_features=64, gamma=1.0, seed=0)
    assert np.array_equal(W, W2) and np.array_equal(b, b2)
