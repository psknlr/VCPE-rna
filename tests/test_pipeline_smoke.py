"""End-to-end smoke test of the P3 pipeline on synthetic data.

Builds two tiny GEARS-format h5ad files with DIFFERENT but overlapping gene
panels and an overlapping set of target genes, then runs training and the
per-axis ablation as subprocesses. No real data assets required.

It exists because the defects in docs/ERRATA.md were all of a kind that a unit
test on a single function cannot see -- a split that leaks, a mask that is not
applied, a metric computed on the wrong tensor, an ablation that varies two
things at once. Those only show up when the pieces run together. In particular
this test asserts that:

  * the gene-grouped split actually holds genes out (E5)
  * the measured-column mask is built and reported (E6)
  * selection happens on the inner split and the test split is scored once (E3)
  * the control baselines run and are reported next to the model
  * the per-axis ablation distinguishes a live conditioning channel from a dead
    one -- the check the P2.3-B report deferred as future work

Skipped unless torch, anndata, scipy and pandas are all importable.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
ad = pytest.importorskip("anndata")
pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_GENES = 60
SYMS = [f"GENE{i:03d}" for i in range(N_GENES)]


def _write_dataset(path, perts, panel, n_cells=6, seed=0):
    r = np.random.default_rng(seed)
    panel_syms = [SYMS[i] for i in panel]
    rows, conds = [], []
    for _ in range(n_cells * 3):
        rows.append(r.gamma(2.0, 1.0, size=len(panel)))
        conds.append("ctrl")
    for g in perts:
        base = r.gamma(2.0, 1.0, size=len(panel))
        for _ in range(n_cells):
            v = base + r.normal(0, 0.15, size=len(panel))
            if SYMS[g] in panel_syms:                       # knock the target down
                v[panel_syms.index(SYMS[g])] *= 0.2
            for off in (1, 2):                              # and its neighbours
                nb = SYMS[(g + off) % N_GENES]
                if nb in panel_syms:
                    v[panel_syms.index(nb)] *= 0.6
            rows.append(np.clip(v, 0, None))
            conds.append(f"{SYMS[g]}+ctrl")
    obs = pd.DataFrame({"condition": conds})
    obs["control"] = (obs["condition"] == "ctrl").astype(int)
    var = pd.DataFrame({"gene_name": panel_syms},
                       index=[f"ENSG{i:08d}" for i in panel])
    ad.AnnData(X=sparse.csr_matrix(np.asarray(rows, dtype=np.float32)),
               obs=obs, var=var).write_h5ad(path)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    torch.save({s: torch.randn(48) for s in SYMS}, d / "esm.pt")
    # overlapping target-gene sets, so a non-grouped split could leak a gene
    _write_dataset(d / "ds1.h5ad", perts=range(0, 30), panel=range(0, 45), seed=1)
    _write_dataset(d / "ds2.h5ad", perts=range(10, 40), panel=range(15, 60), seed=2)
    return d


def _run(args, cwd=ROOT):
    p = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True,
                       text=True, timeout=900)
    assert p.returncode == 0, f"exit {p.returncode}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
    return p.stdout


@pytest.fixture(scope="module")
def trained(synth):
    out = synth / "p3_out"
    log = _run(["src/maprna_p3/train_p3.py",
                "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
                "--esm_table", str(synth / "esm.pt"),
                "--out_dir", str(out), "--epochs", "4", "--n_hvg", "40",
                "--batch_size", "8", "--split_by", "target_gene"])
    return out, log


def test_training_runs_and_reports_the_grouped_split(trained):
    _, log = trained
    assert "[split] by target_gene:" in log
    assert "held-out genes)" in log


def test_measured_mask_is_built_and_reported(trained):
    """E6: unmeasured panel columns must be excluded, and said to be."""
    _, log = trained
    assert "[mask] measured HVG fraction:" in log
    frac = float(log.split("measured HVG fraction: train ")[1].split()[0])
    assert 0.0 < frac < 1.0, "the two panels differ, so the mask must be partial"


def test_selection_uses_the_inner_split_not_the_test_split(trained):
    """E3: the test split must be scored exactly once, after selection."""
    _, log = trained
    assert "inner-val for selection:" in log
    assert "test split is scored once" in log
    assert log.count("held-out results (scored once") == 1


def test_baselines_are_reported_beside_the_model(trained):
    out, log = trained
    for name in ("zero", "train_mean", "knn_esm2", "ridge_esm2"):
        assert name in log, f"baseline {name} missing from the report"
    final = json.loads((out / "final_report.json").read_text())
    assert set(final["baselines"]) == {"zero", "train_mean", "knn_esm2", "ridge_esm2"}
    for m in final["baselines"].values():
        assert np.isfinite(m["pearson_dev"])


def test_checkpoint_records_its_selection_and_split(trained):
    out, _ = trained
    ck = torch.load(out / "ckpt_p3_best_dev.pt", map_location="cpu",
                    weights_only=False)
    assert ck["selection"] == "inner_val_pearson_dev"
    assert ck["split_by"] == "target_gene"
    # E12f: the frozen public ESM2 table must not be inside the checkpoint
    assert "esm_table" not in ck["model_state_dict"]


def test_per_axis_ablation_separates_live_from_dead_channels(synth, trained):
    """The check the P2.3-B report deferred.

    This checkpoint has no RNA encoder and no STRING table, so those two axes are
    *known* to be inert: ablating them must be an exact no-op. The ESM2 axis is
    the only live conditioning channel, so ablating it must change the output.
    A single lumped `ablation_r` cannot make that distinction.
    """
    out, _ = trained
    js = synth / "abl.json"
    _run(["src/maprna_p3/ablate_axes.py",
          "--ckpt", str(out / "ckpt_p3_best_dev.pt"),
          "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
          "--esm_table", str(synth / "esm.pt"), "--n_hvg", "40",
          "--split_by", "target_gene", "--out_json", str(js)])
    rep = json.loads(js.read_text())["report"]

    assert rep["intact"]["r_vs_real"] == pytest.approx(1.0, abs=1e-6)
    # inert axes: exact no-ops
    assert rep["rna_off"]["r_vs_real"] == pytest.approx(1.0, abs=1e-6)
    assert rep["is_nb_off"]["r_vs_real"] == pytest.approx(1.0, abs=1e-6)
    # the live axis moves the output substantially
    assert rep["esm_zero"]["r_vs_real"] < 0.95
    assert rep["esm_shuffle"]["r_vs_real"] < 0.95
    # and the floor is reported, so "conditioning gain" is measurable
    assert "conditioning_gain" in json.loads(js.read_text())["report"]["interpretation"] \
        or "conditioning_gain" in rep["interpretation"]


def test_unmasked_run_inflates_a_baseline_that_has_no_conditioning(synth):
    """E6, measured rather than argued.

    `train_mean` predicts the mean training residual and ignores which gene was
    perturbed, so it carries NO perturbation-specific information. If unmasking
    the never-measured columns lifts its score, the unmasked metric is rewarding
    reproduction of a per-dataset constant -- which is the mechanism, shown
    directly rather than inferred.
    """
    common = ["--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
              "--esm_table", str(synth / "esm.pt"), "--epochs", "4",
              "--n_hvg", "40", "--batch_size", "8", "--split_by", "target_gene"]
    _run(["src/maprna_p3/train_p3.py", "--out_dir", str(synth / "m_on")] + common)
    _run(["src/maprna_p3/train_p3.py", "--out_dir", str(synth / "m_off"),
          "--no_mask"] + common)
    on = json.loads((synth / "m_on" / "final_report.json").read_text())
    off = json.loads((synth / "m_off" / "final_report.json").read_text())

    assert on["measured_fraction"] < 1.0
    assert off["measured_fraction"] == pytest.approx(1.0)
    b_on = on["baselines"]["train_mean"]["pearson_dev"]
    b_off = off["baselines"]["train_mean"]["pearson_dev"]
    assert b_off > b_on, (
        "unmasking must inflate a conditioning-free baseline; got "
        f"masked {b_on:.4f} vs unmasked {b_off:.4f}")


def test_pert_split_warns_that_genes_appear_on_both_sides(synth):
    """E5: the historical split is still available, but must announce the leak."""
    log = _run(["src/maprna_p3/train_p3.py",
                "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
                "--esm_table", str(synth / "esm.pt"),
                "--out_dir", str(synth / "p3_pert"), "--epochs", "1",
                "--n_hvg", "40", "--batch_size", "8", "--split_by", "pert"])
    assert "appear in BOTH splits" in log
    n = int(log.split("WARNING: ")[1].split(" target genes")[0])
    assert n > 0, "the synthetic datasets share target genes, so this must be >0"


def test_panel_is_ranked_on_training_items_by_default(trained):
    """ERRATA E15: the genes the model is scored on must not be chosen using the
    held-out perturbations' expression."""
    _, log = trained
    assert "[hvg] panel ranked on" in log and "TRAINING" in log
    assert "WARNING" not in log.split("[mask]")[0].split("[hvg]")[1]


def test_the_test_visible_panel_is_available_but_says_so_loudly(synth):
    """--hvg_from all reproduces the pre-fix behaviour, so its size can be
    measured; it must never be mistaken for the default."""
    log = _run(["src/maprna_p3/train_p3.py",
                "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
                "--esm_table", str(synth / "esm.pt"),
                "--out_dir", str(synth / "p3_hvg_all"), "--epochs", "1",
                "--n_hvg", "40", "--batch_size", "8",
                "--split_by", "target_gene", "--hvg_from", "all"])
    assert "[hvg] WARNING" in log
    assert "chosen with the test split visible" in log


def test_run_seeds_output_feeds_the_assembler_without_losing_a_column(synth):
    """End to end through the real producers, into the real consumer.

    Three defects of one shape have now been found by hand: the sirnamod
    feature_names/feat_names mismatch, the committed-JSON mismatches in ERRATA
    E11, and the assembler reading `mean` where run_seeds writes
    `mean_difference`. In each case a producer and a consumer disagreed about a
    key and the symptom was a missing number, not an exception. Unit tests with
    hand-written fixtures cannot catch that, because the fixture is written to
    match whichever side the author was looking at.
    """
    out = synth / "seeds_for_assembly"
    _run(["src/maprna_p3/run_seeds.py", "--out_dir", str(out), "--seeds", "0", "1",
          "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
          "--esm_table", str(synth / "esm.pt"), "--epochs", "2", "--n_hvg", "40",
          "--batch_size", "8", "--split_by", "target_gene"])
    summary = out / "seed_summary.json"
    assert summary.exists()

    table = synth / "table.json"
    _run(["tools/assemble_comparison.py", "--reports", str(summary),
          "--out_json", str(table)])
    rep = json.loads(table.read_text())

    labels = {r["label"] for r in rep["rows"]}
    assert "P3 head" in labels
    for name in ("zero", "train_mean", "knn_esm2", "ridge_esm2"):
        assert name in labels, f"{name} did not survive into the table"
        row = [r for r in rep["rows"] if r["label"] == name][0]
        assert row["value"] is not None, f"{name} lost its value"
        assert row["paired_diff"] is not None, \
            f"{name} lost its paired difference -- producer/consumer key mismatch"
        assert row["paired_verdict"], f"{name} lost its verdict"
        assert row["n"] == 2, f"{name} lost the seed count"

    md = (table.with_suffix(".md")).read_text()
    assert "Paired per-seed comparison" in md
    assert "n/a" not in md.split("Paired per-seed comparison")[1], \
        "a column rendered as n/a, which is how a key mismatch looks"
    assert rep["shared_protocol"]["split_by"] == "target_gene"


def test_ablation_quotes_the_shuffle_floor_not_the_zeroed_one(synth, trained):
    """ERRATA E7 follow-up, found by running the ablation on real data.

    Zeroing the target vector takes the input off-distribution, so the model emits
    something degenerate and the floor is flattered -- which inflates the apparent
    conditioning gain. On real data the two floors disagreed badly: 0.006 zeroed
    versus 0.116 shuffled on an intact 0.288, i.e. 98% versus 60% of the score
    attributed to conditioning. Permuting preserves the input statistics and
    destroys only the identity mapping, so it is the honest floor, and the script
    must report that one as the gain.
    """
    out, _ = trained
    rep = synth / "abl_floors.json"
    log = _run(["src/maprna_p3/ablate_axes.py",
                "--ckpt", str(out / "ckpt_p3_best_dev.pt"),
                "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
                "--esm_table", str(synth / "esm.pt"),
                "--n_hvg", "40", "--split_by", "target_gene",
                "--out_json", str(rep)])
    interp = json.loads(rep.read_text())["report"]["interpretation"]
    assert "esm_shuffle" in interp["conditioning_gain_basis"], \
        "the headline gain must be the shuffle-controlled one"
    assert interp["no_conditioning_floor_shuffled"] is not None
    assert "conditioning_gain_vs_zeroed_floor" in interp, \
        "the zeroed-floor gain must still be reported, as an upper bound"
    # the reported gain is measured against the shuffled floor
    assert abs(interp["conditioning_gain"]
               - (interp["intact_pearson_dev"]
                  - interp["no_conditioning_floor_shuffled"])) < 1e-9
    assert "upper bound" in interp["note"] or "upper bound" in log
