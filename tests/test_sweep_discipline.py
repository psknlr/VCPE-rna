"""The feature sweep must be structurally unable to select on the held-out set.

ERRATA E3 -- the reported epoch chosen on the reported slice -- invalidated most of
the withdrawn numbers. A feature search is the same hazard with a different knob:
rank eight designs by their held-out score, report the winner's held-out score, and
the number is a maximum over designs rather than an estimate. The guard cannot be a
promise in a docstring, because the next person to edit the file will not read it.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "sweep_graph_features.py"


def test_the_sweep_never_reads_the_held_out_arrays():
    """dev_te / mask_te / rows_te are the held-out set. Touching any of them here
    would make the ranking a selection on the reported slice."""
    src = SCRIPT.read_text()
    tree = ast.parse(src)
    forbidden = {"dev_te", "mask_te", "rows_te", "test_items"}
    hits = []
    for n in ast.walk(tree):
        # a held-out array can only be reached by subscripting the data dict
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if n.value in forbidden:
                hits.append(n.value)
        if isinstance(n, ast.Name) and n.id in forbidden:
            hits.append(n.id)
        if isinstance(n, ast.Attribute) and n.attr in forbidden:
            hits.append(n.attr)
    assert not hits, (
        f"the sweep references held-out data {sorted(set(hits))}; ranking designs on "
        "the slice they are then reported on is ERRATA E3")


def test_it_ranks_on_inner_validation_and_says_so():
    src = SCRIPT.read_text()
    assert "is_inner_val" in src, "the ranking must use the inner-validation split"
    assert "inner_val" in src.lower()


def test_it_ranks_on_more_than_one_seed_by_default():
    """A single-seed sweep picks whichever design suits one split -- the trap the
    convergence probe fell into, where seed 0 turned out to be the baseline's worst
    split and the only one the head led on."""
    tree = ast.parse(SCRIPT.read_text())
    default = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and getattr(n.args[0], "value", None) == "--seeds"):
            for kw in n.keywords:
                if kw.arg == "default":
                    default = ast.literal_eval(kw.value)
    assert default is not None, "--seeds must have an explicit default"
    assert len(default) >= 3, f"default sweep uses only {default}; rank on >= 3 splits"


def test_it_labels_its_winner_a_candidate_not_a_result():
    """An inner-val winner still has to be scored once on held-out data."""
    src = SCRIPT.read_text()
    assert "CANDIDATE" in src or "candidate" in src
    assert "train_p3" in src, "it must point at the script that scores held-out once"


def test_the_help_text_states_the_discipline():
    p = subprocess.run([sys.executable, str(SCRIPT), "--help"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0
    assert "E3" in p.stdout, "the hazard being guarded against should be named"
