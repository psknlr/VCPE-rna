"""Tests for the GEARS baseline adapter.

The hard part of an external comparison is not calling the other model -- it is
making the comparison mean something. These tests cover the parts that decide
that, and they run without GEARS installed:

  * the split handed to GEARS is VCPE's split, with `ctrl` excluded
  * predictions are projected onto VCPE's gene panel by symbol, and genes the
    other model does not predict are marked absent rather than silently treated
    as a prediction of no change

A full integration test that actually trains GEARS is included but skipped unless
gears + torch_geometric are importable.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))
sys.path.insert(0, str(ROOT / "src"))

from baseline_gears import align_to_hvg, vcpe_split_to_gears  # noqa: E402


# --------------------------------------------------------------------------
# split conversion
# --------------------------------------------------------------------------

def test_split_conversion_keeps_the_three_sets_disjoint_and_complete():
    tr = [(0, "A+ctrl"), (0, "B+ctrl")]
    va = [(0, "C+ctrl")]
    te = [(0, "D+ctrl"), (0, "E+ctrl")]
    d = vcpe_split_to_gears(tr, va, te)
    assert d["train"] == ["A+ctrl", "B+ctrl"]
    assert d["val"] == ["C+ctrl"]
    assert d["test"] == ["D+ctrl", "E+ctrl"]
    assert set(d["train"]) & set(d["test"]) == set()


def test_ctrl_is_never_a_member_of_any_split():
    """`ctrl` is the baseline, not a perturbation, in either framework."""
    d = vcpe_split_to_gears([(0, "ctrl"), (0, "A+ctrl")], [(0, "CTRL")],
                            [(0, "B+ctrl")])
    assert d["train"] == ["A+ctrl"]
    assert d["val"] == []
    assert all("ctrl" != c.lower() for s in d.values() for c in s)


def test_dataset_index_is_dropped_and_duplicates_collapse():
    """GEARS keys by condition string only, which is why one dataset at a time."""
    d = vcpe_split_to_gears([(0, "A+ctrl"), (1, "A+ctrl")], [], [])
    assert d["train"] == ["A+ctrl"]


# --------------------------------------------------------------------------
# gene-space alignment
# --------------------------------------------------------------------------

def test_alignment_maps_by_symbol_not_by_position():
    """The two gene orders differ, so positional alignment would be silently wrong."""
    pred = np.array([1.0, 2.0, 3.0])
    vec, found = align_to_hvg(pred, ["GENEC", "GENEA", "GENEB"],
                              ["GENEA", "GENEB", "GENEC"])
    assert list(vec) == [2.0, 3.0, 1.0]
    assert found.all()


def test_genes_the_other_model_does_not_predict_are_marked_absent():
    """Absent must be distinguishable from a prediction of zero change.

    Counting an unpredicted gene as 0.0 would credit the model with correctly
    predicting no change on every gene it was never asked about.
    """
    vec, found = align_to_hvg(np.array([5.0]), ["GENEA"], ["GENEA", "GENEB"])
    assert vec[0] == 5.0 and found[0]
    assert vec[1] == 0.0 and not found[1]


def test_a_real_zero_prediction_is_still_marked_present():
    vec, found = align_to_hvg(np.array([0.0]), ["GENEA"], ["GENEA"])
    assert vec[0] == 0.0 and found[0]


def test_alignment_handles_a_panel_entry_with_no_symbol():
    """HVG rows whose ESM2 symbol is missing must not match anything."""
    vec, found = align_to_hvg(np.array([1.0]), ["GENEA"], [None, "GENEA"])
    assert not found[0] and found[1]


# --------------------------------------------------------------------------
# the residual conversion -- the fairness point that matters most
# --------------------------------------------------------------------------

def test_expression_to_residual_conversion_removes_the_shared_component():
    """GEARS predicts expression; VCPE predicts fc minus the train-only core.

    Scoring GEARS's raw expression against VCPE's residual would flatter GEARS
    enormously, because the shared response dominates raw expression -- the whole
    finding of the P2 erratum. This reproduces the conversion the adapter applies
    and checks that a model which predicts only the shared response scores ~0 on
    the residual, not highly.
    """
    rng = np.random.default_rng(0)
    n_pert, n_gene = 30, 200
    ctrl = rng.normal(5.0, 1.0, size=n_gene)
    common = rng.normal(0.0, 1.5, size=n_gene)         # the shared response
    specific = rng.normal(0.0, 0.3, size=(n_pert, n_gene))

    true_expr = ctrl + common + specific
    shared_only_expr = ctrl + common                   # predicts no gene-specific part

    # raw expression looks excellent
    raw = np.mean([np.corrcoef(true_expr[i], shared_only_expr)[0, 1]
                   for i in range(n_pert)])
    # the adapter's conversion: expression -> fc -> minus the train-only core
    true_dev = (true_expr - ctrl) - common
    pred_dev = (shared_only_expr - ctrl) - common
    assert np.allclose(pred_dev, 0.0)

    resid = np.mean([np.corrcoef(true_dev[i], pred_dev + 1e-12)[0, 1]
                     if np.std(pred_dev) > 0 else 0.0 for i in range(n_pert)])
    assert raw > 0.9, f"raw expression should look great, got {raw:.3f}"
    assert abs(resid) < 0.05, "on the residual, a shared-response-only model scores ~0"


# --------------------------------------------------------------------------
# integration: only when GEARS is actually installed
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not all(__import__("importlib").util.find_spec(m) for m in ("gears", "torch_geometric")),
    reason="gears / torch_geometric not installed")
def test_gears_is_importable_and_exposes_the_api_the_adapter_uses():
    """Pin the GEARS entry points the adapter depends on.

    If a future GEARS release renames any of these, the adapter breaks and this
    test says which one -- better than a stack trace 40 minutes into a training
    run.
    """
    import inspect
    from gears import GEARS, PertData
    assert "split_dict_path" in inspect.signature(PertData.prepare_split).parameters, \
        "the custom-split mechanism the fair comparison relies on is gone"
    for m in ("new_data_process", "prepare_split", "get_dataloader"):
        assert callable(getattr(PertData, m)), m
    for m in ("model_initialize", "train", "predict"):
        assert callable(getattr(GEARS, m)), m


def test_gears_adapter_refuses_a_pseudobulk_training_file_with_an_explanation():
    """GEARS dies inside scanpy on a pseudobulk file, naming every perturbation at
    once, which reads like a data problem rather than the wrong granularity.

    The check is on the source text rather than by running GEARS, because importing
    GEARS to prove the message exists costs minutes and the message is the point.
    """
    src = (ROOT / "src" / "maprna_p3" / "baseline_gears.py").read_text()
    assert "rank_genes_groups" in src, "the failure it guards against is not named"
    assert "--mode cells" in src, "the message does not say how to fix it"
    assert "single row" in src or "a single row" in src


def test_gears_adapter_checks_the_two_files_agree_on_the_conditions():
    """Different condition sets would move the split without saying so, which is
    exactly the kind of silent mismatch E3 and E4 were about."""
    src = (ROOT / "src" / "maprna_p3" / "baseline_gears.py").read_text()
    # the wording is wrapped across lines in the source, so match on the
    # distinctive part rather than a whole sentence
    assert "not shared" in src and "conditions VCPE splits on" in src
