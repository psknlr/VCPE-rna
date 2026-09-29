"""Tests for the CPA baseline adapter's own logic.

CPA cannot be imported in this environment -- it pins `torch<2.0.0` while this
repository requires `torch>=2.1`, which is why the adapter is split into stages
that run in different environments (docs/ERRATA.md E14b). So these tests cover the
parts that decide whether the comparison is valid and that do not need CPA:

  * the perturbation-label translation, which was the difference between CPA
    reading "one perturbed gene" and CPA reading "a combination of MYC with the
    control"
  * the export/score file interface between the stages

The `run` stage is exercised by actually running it (see the errata); it cannot be
unit-tested here because importing CPA would break the environment.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "maprna_p3" / "baseline_cpa.py"
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))


# --------------------------------------------------------------------------
# label translation -- reproduced here because it lives inside stage_run, which
# cannot be imported without CPA
# --------------------------------------------------------------------------

def to_cpa_label(c):
    c = str(c)
    if c.lower() == "ctrl":
        return "ctrl"
    parts = [x for x in c.split("+") if x.lower() != "ctrl"]
    return "+".join(parts) if parts else "ctrl"


def test_single_gene_label_loses_the_ctrl_suffix():
    """GEARS writes "GENE+ctrl" for a single perturbation.

    CPA splits its perturbation key on "+" and counts every part, so the raw
    label reads as a two-way combination of the gene with the control. That makes
    the combination lengths ragged against plain "ctrl" rows and fails inside
    np.vstack, and it would also misrepresent the experiment.
    """
    assert to_cpa_label("MYC+ctrl") == "MYC"


def test_control_stays_control():
    assert to_cpa_label("ctrl") == "ctrl"
    assert to_cpa_label("CTRL") == "ctrl"


def test_a_genuine_combination_is_preserved():
    """A real double perturbation must keep both genes."""
    assert to_cpa_label("MYC+TP53") == "MYC+TP53"


def test_a_label_that_is_only_controls_collapses_to_ctrl():
    assert to_cpa_label("ctrl+ctrl") == "ctrl"


def test_translation_is_idempotent():
    """Applying it twice must not change anything, since obs and the split sets
    are both translated and could otherwise drift apart."""
    for c in ("MYC+ctrl", "ctrl", "MYC+TP53"):
        assert to_cpa_label(to_cpa_label(c)) == to_cpa_label(c)


def test_max_combination_length_is_derived_from_the_data():
    """max_comb_len must cover the longest real combination, or vstack fails."""
    labels = {"ctrl", "MYC", "TP53", "MYC+TP53"}
    max_comb = max(1, max(len(c.split("+")) for c in labels if c != "ctrl"))
    assert max_comb == 2
    singles_only = {"ctrl", "MYC", "TP53"}
    assert max(1, max(len(c.split("+")) for c in singles_only if c != "ctrl")) == 1


# --------------------------------------------------------------------------
# the stage interface
# --------------------------------------------------------------------------

def test_a_stage_is_required():
    """Running with no stage must explain the order rather than doing something."""
    p = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                       timeout=120)
    assert p.returncode != 0
    assert "export" in (p.stdout + p.stderr) and "score" in (p.stdout + p.stderr)


def test_export_refuses_more_than_one_dataset():
    """CPA is set up per AnnData; a multi-dataset VCPE run has no counterpart."""
    p = subprocess.run(
        [sys.executable, str(SCRIPT), "export", "--work", "/tmp/nonexistent_cpa",
         "--data_dirs", "a.h5ad", "b.h5ad", "--esm_table", "x.pt"],
        capture_output=True, text=True, timeout=120)
    assert p.returncode != 0
    assert "exactly one h5ad" in (p.stdout + p.stderr)


def test_score_reads_the_interface_the_export_stage_writes(tmp_path):
    """The two stages run in different environments, so the file contract between
    them is the only thing holding the comparison together."""
    # >= 10 genes: per_item_correlation refuses to score an item with
    # fewer measured features, which is the guard working, not a bug
    n_pert, n_gene = 3, 20
    syms = [f"G{i}" for i in range(n_gene)]
    common = np.zeros(n_gene, dtype=np.float32)
    dev = np.arange(n_pert * n_gene, dtype=np.float32).reshape(n_pert, n_gene)

    (tmp_path / "meta.json").write_text(json.dumps(dict(
        h5ad="unused.h5ad", split_by="target_gene", seed=0, n_hvg=n_gene,
        hvg_symbols=syms, train_conditions=[], valid_conditions=[],
        test_conditions=[f"P{i}+ctrl" for i in range(n_pert)],
        test_dataset_idx=[0] * n_pert)))
    np.savez_compressed(tmp_path / "vcpe.npz", hvg_rows=np.arange(n_gene),
                        dev_te=dev, mask_te=np.ones((n_pert, n_gene), bool),
                        common_fc=common, ctrl_feat_rows=np.zeros(n_pert, int))
    # a perfect predictor: expression = dev + ctrl, so dev round-trips exactly
    ctrl = np.full(n_gene, 7.0, dtype=np.float32)
    np.savez_compressed(tmp_path / "cpa_pred.npz",
                        pred_expr=(dev + ctrl).astype(np.float32),
                        ctrl_mean=ctrl,
                        gene_names=np.array(syms, dtype=object),
                        n_cells_per_condition=np.full(n_pert, 6))

    out = tmp_path / "rep.json"
    p = subprocess.run([sys.executable, str(SCRIPT), "score", "--work", str(tmp_path),
                        "--out_json", str(out)], capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads(out.read_text())
    assert rep["cpa"]["pearson_dev"] == pytest.approx(1.0), \
        "a predictor that reproduces dev exactly must score 1.0"
    assert rep["intersected_mask_fraction"] == pytest.approx(1.0)
    assert rep["fairness"]["not_controlled"], "the caveats must be carried through"
    assert "torch<2" in rep["environment_note"], \
        "the environment split is the reason this is a three-stage script"


def test_score_marks_genes_cpa_does_not_cover(tmp_path):
    """A gene the external model never predicted must not be scored as 0.0."""
    n_pert, n_gene = 2, 24
    syms = [f"G{i}" for i in range(n_gene)]
    dev = np.ones((n_pert, n_gene), dtype=np.float32)
    (tmp_path / "meta.json").write_text(json.dumps(dict(
        h5ad="u.h5ad", split_by="target_gene", seed=0, n_hvg=n_gene,
        hvg_symbols=syms, train_conditions=[], valid_conditions=[],
        test_conditions=["A+ctrl", "B+ctrl"], test_dataset_idx=[0, 0])))
    np.savez_compressed(tmp_path / "vcpe.npz", hvg_rows=np.arange(n_gene),
                        dev_te=dev, mask_te=np.ones((n_pert, n_gene), bool),
                        common_fc=np.zeros(n_gene, np.float32),
                        ctrl_feat_rows=np.zeros(n_pert, int))
    # CPA covers only the first two genes
    np.savez_compressed(tmp_path / "cpa_pred.npz",
                        pred_expr=np.ones((n_pert, 12), np.float32),
                        ctrl_mean=np.zeros(12, np.float32),
                        gene_names=np.array(syms[:12], dtype=object),
                        n_cells_per_condition=np.full(n_pert, 3))
    out = tmp_path / "rep.json"
    p = subprocess.run([sys.executable, str(SCRIPT), "score", "--work", str(tmp_path),
                        "--out_json", str(out)], capture_output=True, text=True,
                       timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads(out.read_text())
    assert rep["cpa_coverage_fraction"] == pytest.approx(0.5)
    assert rep["intersected_mask_fraction"] == pytest.approx(0.5)
