"""Tests for the shared external-baseline comparison machinery.

This module is the single path every external baseline is scored through. It
exists because ERRATA E1 and E2 both came from each script re-implementing its own
metrics until the definitions drifted -- one of them ended up correlating a feature
instead of a label. Re-implementing the *comparison* per baseline would reproduce
that one level up, so these tests pin the shared behaviour rather than any one
adapter's.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))

from baseline_common import (  # noqa: E402
    UNCONTROLLED_FACTORS, align_to_panel, expression_to_dev, fairness_block,
    hvg_symbols, inner_split_items, intersect_masks, score)


# --------------------------------------------------------------------------
# gene-space alignment
# --------------------------------------------------------------------------

def test_alignment_is_by_symbol_not_position():
    """The two gene orders generally differ, so positional alignment is wrong."""
    vec, found = align_to_panel(np.array([1.0, 2.0, 3.0]),
                               ["C", "A", "B"], ["A", "B", "C"])
    assert list(vec) == [2.0, 3.0, 1.0]
    assert found.all()


def test_uncovered_genes_are_absent_not_zero():
    """Absent must differ from a prediction of no change.

    Counting an uncovered gene as 0.0 credits the model with correctly predicting
    no change on a gene it was never asked about.
    """
    vec, found = align_to_panel(np.array([5.0]), ["A"], ["A", "B"])
    assert (vec[0], bool(found[0])) == (5.0, True)
    assert (vec[1], bool(found[1])) == (0.0, False)


def test_a_genuine_zero_is_still_present():
    _, found = align_to_panel(np.array([0.0]), ["A"], ["A"])
    assert found[0]


def test_panel_entries_without_a_symbol_never_match():
    """HVG rows whose ESM2 symbol is missing must not be matched to anything."""
    _, found = align_to_panel(np.array([1.0]), ["A"], [None, "A"])
    assert not found[0] and found[1]


def test_alignment_does_not_match_the_string_none():
    """A literal 'None' gene name must not be matched by a missing symbol."""
    _, found = align_to_panel(np.array([1.0]), ["None"], [None])
    assert not found[0]


# --------------------------------------------------------------------------
# the residual conversion -- the fairness point that matters most
# --------------------------------------------------------------------------

def test_expression_to_dev_is_the_documented_formula():
    p, c, k = np.array([3.0, 4.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5])
    assert np.allclose(expression_to_dev(p, c, k), [1.5, 2.5])


def test_a_shared_response_only_model_scores_high_on_expression_and_zero_on_residual():
    """Why the conversion is not optional.

    An external model that predicts only the response shared across perturbations
    looks excellent on raw expression and has, correctly, no gene-specific signal
    once the shared core is removed. Scoring raw expression against VCPE's
    residual would hand such a model a large score for free -- which is the P2
    erratum applied to a baseline comparison.
    """
    rng = np.random.default_rng(0)
    n_pert, n_gene = 40, 200
    ctrl = rng.normal(5.0, 1.0, size=n_gene)
    common = rng.normal(0.0, 1.5, size=n_gene)
    true_expr = ctrl + common + rng.normal(0.0, 0.3, size=(n_pert, n_gene))
    shared_only = ctrl + common                      # no perturbation-specific part

    raw = float(np.mean([np.corrcoef(true_expr[i], shared_only)[0, 1]
                         for i in range(n_pert)]))
    assert raw > 0.9, f"raw expression should look excellent, got {raw:.3f}"

    pred_dev = expression_to_dev(np.tile(shared_only, (n_pert, 1)),
                                 np.tile(ctrl, (n_pert, 1)),
                                 common)
    assert np.allclose(pred_dev, 0.0), "the shared-only model has no residual left"


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------

def test_masks_intersect_so_only_mutually_defined_entries_are_scored():
    a = np.array([[True, True, False]])
    b = np.array([[True, False, True]])
    assert intersect_masks(a, b).tolist() == [[True, False, False]]


def test_score_respects_the_mask():
    """A column outside the mask must not influence the result."""
    rng = np.random.default_rng(1)
    true = rng.normal(size=(20, 60))
    pred = true.copy()
    pred[:, 30:] = rng.normal(size=(20, 30)) * 50      # garbage, but masked out
    mask = np.zeros((20, 60), bool)
    mask[:, :30] = True
    assert score(true, pred, mask, n_boot=50)["pearson_dev"] == pytest.approx(1.0)


def test_score_reports_the_estimator_it_used():
    """A number without its estimator name is how E4 happened."""
    rng = np.random.default_rng(2)
    true = rng.normal(size=(20, 40))
    s = score(true, true.copy(), np.ones((20, 40), bool), n_boot=50)
    est = s["pearson_dev_estimator"]
    assert "within_perturbation" in est, f"estimator name is uninformative: {est!r}"
    assert "pearson_dev_pooled" in s and "pearson_dev" in s


def test_score_exposes_both_estimators_separately():
    """They answer different questions and must never be conflated."""
    rng = np.random.default_rng(3)
    level = rng.normal(scale=5.0, size=30)
    true = level[:, None] + rng.normal(size=(30, 100))
    pred = level[:, None] + rng.normal(size=(30, 100))
    s = score(true, pred, np.ones((30, 100), bool), n_boot=50)
    assert abs(s["pearson_dev"]) < 0.1          # no within-perturbation signal
    assert s["pearson_dev_pooled"] > 0.7        # but the level is predicted


# --------------------------------------------------------------------------
# split + panel helpers, and the fairness statement
# --------------------------------------------------------------------------

def test_inner_split_items_matches_what_train_p3_trains_and_scores_on():
    data = {"train_items": [(0, "A"), (0, "B"), (0, "C")],
            "is_inner_val": np.array([False, True, False]),
            "test_items": [(0, "D")]}
    tr, va, te = inner_split_items(data)
    assert tr == [(0, "A"), (0, "C")]
    assert va == [(0, "B")]
    assert te == [(0, "D")]


def test_hvg_symbols_inverts_the_symbol_table():
    data = {"sym2row": {"GENEA": 5, "GENEB": 7}, "hvg_rows": [7, 5, 9]}
    assert hvg_symbols(data) == ["GENEB", "GENEA", None]


def test_fairness_block_always_states_what_is_not_controlled():
    """A comparison that omits its uncontrolled factors is a misleading one."""
    f = fairness_block()
    assert f["controlled"] and f["not_controlled"]
    joined = " ".join(f["not_controlled"]).lower()
    assert "tuning" in joined, "the tuning asymmetry must be stated"
    assert "capacity" in joined or "larger" in joined
    assert len(f["not_controlled"]) >= len(UNCONTROLLED_FACTORS)


def test_fairness_block_carries_adapter_specific_caveats_too():
    f = fairness_block(extra=["this baseline needed a version workaround"])
    assert any("workaround" in c for c in f["not_controlled"])
    assert len(f["not_controlled"]) == len(UNCONTROLLED_FACTORS) + 1
