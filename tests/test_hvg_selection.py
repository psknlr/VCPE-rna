"""The response panel must be chosen without looking at the held-out perturbations.

`make_hvg_list` ranks genes by variance and used to be called before the split, so
the genes the model was trained and scored on were selected using the test items'
expression (docs/ERRATA.md E15). This is feature selection on the full data rather
than a per-item label leak, which makes it the mildest defect in that document --
but "the held-out genes are genuinely unseen" cannot be true of the split and
false of the panel.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p1"))

from ds_knockdown import make_hvg_list  # noqa: E402


def kd_with(n_pert, n_gene, varying_gene, varying_rows):
    """A one-dataset kd whose only variable gene varies on `varying_rows` alone."""
    X = np.ones((n_pert, n_gene), dtype=np.float32)
    X[varying_rows, varying_gene] += np.arange(1, len(varying_rows) + 1) * 10.0
    return dict(X_pert=[X], row_of_gene=[np.arange(n_gene, dtype=np.int64)])


def test_a_gene_that_varies_only_among_held_out_items_is_not_selected():
    """The leak, stated as the smallest case that shows it.

    Gene 3 is constant across the training perturbations and varies only across
    the held-out ones. Ranking on everything puts it in the panel; ranking on the
    training items cannot see it.
    """
    n_pert, n_gene = 10, 6
    test_rows = [8, 9]
    kd = kd_with(n_pert, n_gene, varying_gene=3, varying_rows=test_rows)
    train_rows = np.array([i for i in range(n_pert) if i not in test_rows])

    leaky = make_hvg_list(kd, n_hvg=1)
    clean = make_hvg_list(kd, n_hvg=1, keep_rows=[train_rows])
    assert leaky.tolist() == [3], "gene 3 is the only variable gene overall"
    assert 3 not in clean.tolist(), \
        "a gene that varies only among the held-out perturbations entered the panel"


def test_a_gene_that_varies_in_training_is_still_selected():
    """The fix must not simply drop genes."""
    kd = kd_with(10, 6, varying_gene=2, varying_rows=[0, 1, 2, 3])
    train_rows = np.arange(8)
    assert make_hvg_list(kd, n_hvg=1, keep_rows=[train_rows]).tolist() == [2]


def test_ranking_on_everything_is_the_default_only_for_reproducing_the_old_runs():
    """keep_rows=None keeps the historical behaviour, so the two can be compared."""
    kd = kd_with(10, 6, varying_gene=4, varying_rows=[9])
    assert make_hvg_list(kd, n_hvg=1).tolist() == [4]


def test_a_dataset_with_fewer_than_two_training_items_is_skipped_not_ranked():
    """Variance over one item is zero for every gene, which would otherwise let an
    all-zero ranking from one dataset dilute a real one from another."""
    X = np.ones((3, 4), dtype=np.float32)
    X[:, 1] = [1.0, 5.0, 9.0]
    kd = dict(X_pert=[X, X.copy()],
              row_of_gene=[np.arange(4, dtype=np.int64), np.arange(4, dtype=np.int64)])
    # dataset 1 contributes a single training item -> must not contribute variance
    got = make_hvg_list(kd, n_hvg=1, keep_rows=[np.array([0, 1, 2]), np.array([0])])
    assert got.tolist() == [1]


def test_no_training_items_at_all_is_an_empty_ranking_rather_than_a_wrong_one():
    kd = kd_with(5, 4, varying_gene=1, varying_rows=[4])
    assert len(make_hvg_list(kd, n_hvg=2, keep_rows=[np.array([], dtype=int)])) == 0
