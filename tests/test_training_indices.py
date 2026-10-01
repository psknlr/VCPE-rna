"""ERRATA E17: the inner-validation slice must not be in the training batches."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))

import train_p3  # noqa: E402


def _data():
    iv = np.zeros(20, dtype=bool)
    iv[[1, 4, 9, 15]] = True
    return dict(train_items=list(range(20)), is_inner_val=iv)


def test_default_holds_the_inner_validation_slice_out_of_training():
    d = _data()
    idx = train_p3.training_indices(d)
    assert set(idx).isdisjoint(np.flatnonzero(d["is_inner_val"]))
    assert len(idx) == 16


def test_legacy_flag_reproduces_the_old_behaviour_exactly():
    d = _data()
    assert list(train_p3.training_indices(d, inner_val_in_training=True)) == \
        list(range(20))


def test_an_empty_inner_split_is_refused():
    d = _data()
    d["is_inner_val"][:] = False
    with pytest.raises(AssertionError):
        train_p3.training_indices(d)
