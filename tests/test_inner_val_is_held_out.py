"""The inner-validation slice must be held out of the gradient, structurally.

ERRATA E16. `--inner_val_frac` was added to fix E3 -- an epoch selected on the
slice being reported -- by carving a selection slice out of the training items.
But `train_p3.py` went on building its batches from *every* training item, so the
selection slice was training data. Two consequences, both invisible in a report:

  * the selection curve is a training curve. It rises for as long as the budget
    lasts, which is exactly what every long run recorded (epochs 78/80 at an
    80-epoch budget, 150/150 at 150), and that was read as "the model has not
    converged" rather than "this is not a validation curve".
  * the head saw ~18% more training items than GEARS and CPA, which
    `baseline_common.inner_split_items` hands the inner-TRAIN items -- in a
    module whose stated purpose is identical footing.

A docstring cannot hold this: the defect is one index set away at all times. The
checks here are deliberately torch-free so they run in the fast CI job alongside
the other metric-definition regressions.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAIN_P3 = ROOT / "src" / "maprna_p3" / "train_p3.py"
RUN_SEEDS = ROOT / "src" / "maprna_p3" / "run_seeds.py"


def _main_body():
    tree = ast.parse(TRAIN_P3.read_text())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "main":
            return n
    raise AssertionError("train_p3.main() not found")


def _assign_targets(node):
    """{name: source-ish repr of value} for every assignment under `node`."""
    out = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ast.dump(n.value)
    return out


def test_the_batch_permutation_is_drawn_from_the_fit_set_not_every_item():
    """`perm` must index the fit set. `torch.randperm(len(rows_tr))` would walk
    every training item, inner-validation ones included."""
    assigns = _assign_targets(_main_body())
    assert "perm" in assigns, "train_p3.main() no longer builds a batch permutation"
    perm = assigns["perm"]
    assert "fit_pos" in perm, (
        "the batch permutation does not go through the fit set, so the "
        "inner-validation items are being trained on (ERRATA E16). Got: " + perm)


def test_the_fit_set_is_the_complement_of_the_inner_validation_slice():
    src = TRAIN_P3.read_text()
    assert "flatnonzero(~is_val)" in src, (
        "the fit set must be derived as the complement of is_inner_val, so the "
        "two cannot drift apart")


def test_the_disjointness_is_asserted_at_runtime_not_promised_in_a_comment():
    """Every other guard in this repository is mechanical; this one must be too."""
    body = _main_body()
    asserts = [n for n in ast.walk(body) if isinstance(n, ast.Assert)]
    srcs = [ast.dump(a.test) for a in asserts]
    assert any("is_val" in s and "fit_pos" in s for s in srcs), (
        "no runtime assertion that the gradient set and the selection set are "
        "disjoint")


def test_the_old_behaviour_is_reachable_for_measurement_but_not_the_default():
    """The repository's house rule for a protocol defect: keep the broken path
    behind a flag so its size can be measured (cf. --no_mask for E6,
    --hvg_from all for E15), and default to the corrected one."""
    tree = ast.parse(TRAIN_P3.read_text())
    flag = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and getattr(n.args[0], "value", None) == "--train_on_inner_val"):
            flag = {kw.arg: ast.literal_eval(kw.value) for kw in n.keywords
                    if kw.arg in ("action", "default")}
    assert flag is not None, "--train_on_inner_val is not declared"
    assert flag.get("action") == "store_true" and flag.get("default") is False, (
        "training on the selection slice must be opt-in, not the default")


@pytest.mark.parametrize("field", [
    "trained_on_inner_val",   # which rule produced this number
    "n_fit_items",            # how many items every row of the table was fitted on
    "n_inner_val_items",
    "selection_hit_budget",   # whether the budget, not the data, ended the run
])
def test_the_report_states_the_fit_rule(field):
    """ERRATA E11: a number that does not carry its protocol cannot be compared
    with another number. The fit set is part of the protocol."""
    assert f'"{field}"' in TRAIN_P3.read_text(), (
        f"final_report.json does not record {field}")


def test_seeds_cannot_be_aggregated_across_different_fit_rules():
    """run_seeds refuses to average seeds run under different protocols; the fit
    rule changes the number, so it belongs in that check."""
    tree = ast.parse(RUN_SEEDS.read_text())
    keys = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "PROTOCOL_KEYS"
                for t in n.targets):
            keys = ast.literal_eval(n.value)
    assert keys is not None, "PROTOCOL_KEYS not found in run_seeds.py"
    assert "train_on_inner_val" in keys, (
        "seeds trained under different fit rules would be averaged together")


def test_the_controls_are_fitted_on_the_same_items_as_the_head():
    """A control fitted on more items than the model it controls for is E16 with
    the sign reversed."""
    src = TRAIN_P3.read_text()
    call = src.split("base_preds = run_all(")[1].split(")\n")[0]
    for arg in ('data["dev_tr"][fit_pos]', 'data["rows_tr"][fit_pos]'):
        assert arg in call, (
            f"run_all() is not restricted to the fit set ({arg} missing), so the "
            "baselines see items the head's gradient did not")
