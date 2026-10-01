"""Every script that builds the comparison data must expose the same protocol.

ERRATA E1 and E2 both came from per-file reimplementation drifting apart, and E15
showed the same shape one level up: a flag that changes which genes are scored was
added to `train_p3.py`, and the four other scripts that call `build_dev_data`
carried on building an `argparse.Namespace` without it. Three of them are the
external-baseline adapters, whose entire claim is that they run on VCPE's split
and panel -- a panel they could no longer be told to match. The fourth crashed.

An unresolved-name check cannot catch this: the missing name is an attribute on a
Namespace, which exists only at runtime. So it is checked structurally here.

These flags are the protocol: they decide which items are held out, which genes
are scored, and how many. A number quoted without them is not comparable to
another number, which is what E3 and E4 were about.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# flags that change WHICH items are held out or WHICH genes are scored
PROTOCOL_FLAGS = {"--n_hvg", "--split_by", "--hvg_from", "--test_frac", "--seed"}

CALLERS = ["src/maprna_p3/train_p3.py", "src/maprna_p3/ablate_axes.py",
           "src/maprna_p3/eval_fair.py", "src/maprna_p3/baseline_gears.py",
           "src/maprna_p3/baseline_cpa.py", "tools/sweep_graph_features.py"]

# A caller may express the split seed singly (--seed) or plurally (--seeds, when it
# deliberately ranks over several splits). Either states which splits produced the
# number, which is what the requirement is for.
SEED_EQUIVALENTS = {"--seed", "--seeds"}


def declared_flags(path):
    tree = ast.parse((ROOT / path).read_text())
    out = set()
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add_argument" and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            out.add(n.args[0].value)
    return out


def calls_build_dev_data(path):
    src = (ROOT / path).read_text()
    return "build_dev_data(" in src.replace("def build_dev_data(", "")


@pytest.mark.parametrize("path", CALLERS)
def test_every_build_dev_data_caller_exposes_the_protocol_flags(path):
    assert calls_build_dev_data(path), f"{path} no longer calls build_dev_data"
    have = declared_flags(path)
    need = set(PROTOCOL_FLAGS)
    if have & SEED_EQUIVALENTS:
        need.discard("--seed")
    missing = sorted(need - have)
    assert not missing, (
        f"{path} calls build_dev_data but cannot be told {missing}. An external "
        "baseline that cannot be told the panel and split cannot be claimed to "
        "share them.")


def test_the_caller_list_is_still_complete():
    """A new caller must be added to this test rather than quietly skipped."""
    found = []
    for f in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tools").rglob("*.py")):
        rel = str(f.relative_to(ROOT))
        if rel.endswith("train_p3.py") or "build_dev_data" not in f.read_text():
            continue
        found.append(rel)
    extra = sorted(set(found) - set(CALLERS))
    assert not extra, f"new build_dev_data caller(s) not covered here: {extra}"
