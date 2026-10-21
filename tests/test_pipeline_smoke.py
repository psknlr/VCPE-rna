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
    # neighbor_prior is absent without a STRING table on purpose: with no graph it
    # is identical to train_mean, and a duplicate row reads like a second,
    # independent control agreeing with the first.
    assert set(final["baselines"]) == {
        "zero", "train_mean", "knn_esm2", "ridge_esm2",
        # the same estimators with their one knob chosen on inner validation,
        # plus the non-linear rung between a linear map and the head
        "ridge_esm2_sel", "knn_esm2_sel", "rff_ridge_esm2_sel"}
    for m in final["baselines"].values():
        assert np.isfinite(m["pearson_dev"])


def test_every_control_row_carries_its_hyperparameters_and_how_they_were_chosen(trained):
    """A control quoted without its hyperparameter is as uncomparable as a score
    quoted without its estimator: alpha=1.0 and a selected alpha are different
    controls, and the gap between them is part of what the table shows."""
    out, log = trained
    final = json.loads((out / "final_report.json").read_text())
    b = final["baselines"]
    assert b["ridge_esm2"]["hyperparameters"] == {"alpha": 1.0,
                                                  "chosen_by": "fixed default"}
    assert b["knn_esm2"]["hyperparameters"] == {"k": 10,
                                                "chosen_by": "fixed default"}
    for name in ("ridge_esm2_sel", "knn_esm2_sel", "rff_ridge_esm2_sel"):
        assert b[name]["hyperparameters"]["chosen_by"] == "inner_val"

    sel = final["control_selection"]
    assert sel["selected_on"] == "inner_val"
    assert sel["n_val_items"] == final["n_inner_val_items"], \
        "the controls must be selected on the slice that selects the head's epoch"
    assert sel["n_fit_items"] == final["n_fit_items"]
    assert len(sel["grids"]["ridge_alpha"]) >= 5
    assert "held-out set not consulted" in log


def test_each_control_carries_a_paired_item_level_interval_against_the_model(trained):
    """Every comparison in this repository was settled by a normal approximation
    over 3 seeds, which at n=3 is wide almost regardless of the data. The
    item-level interval is a different and much larger sample -- and it must say
    what it does not cover, or a reader will take a three-seed tie as resolved."""
    out, _ = trained
    final = json.loads((out / "final_report.json").read_text())
    for name, m in final["baselines"].items():
        pb = m["paired_vs_model_items"]
        assert pb["resampled"] == "items within one split"
        assert "run_seeds" in pb["does_not_cover"]
        # the reported difference must be the one the two scored numbers imply
        assert pb["difference"] == pytest.approx(
            final["pearson_dev"] - m["pearson_dev"], abs=1e-9), \
            f"{name}'s paired difference disagrees with its own row"


def test_secondary_metrics_are_reported_with_their_chance_levels(trained):
    """ERRATA E2 was a metric whose chance level was misscaled, so chance read
    0.25. The chance level now travels with the number."""
    out, _ = trained
    final = json.loads((out / "final_report.json").read_text())
    rows = [final] + list(final["baselines"].values())
    for r in rows:
        assert set(r["secondary_metric_chance_levels"]) == {
            "spearman_dev", "enrichment_top5pct_dev",
            "direction_agreement_top10pct_dev"}
        assert r["secondary_metric_chance_levels"]["enrichment_top5pct_dev"] == 1.0
        for k in r["secondary_metric_chance_levels"]:
            assert k in r, f"{k} has a chance level but no value"
    # the floor baseline must sit at chance on the metrics that have one
    zero = final["baselines"]["zero"]
    assert zero["spearman_dev"] == pytest.approx(0.0)


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


def test_the_selection_slice_is_not_in_the_gradient_and_the_report_says_so(trained):
    """ERRATA E16, measured through the real producer.

    The inner-validation slice selects the epoch, so it must not be fitted. The
    accounting has to add up in the report as well as in the loop, because the
    footing claim made in Table D -- that the head and the external baselines
    share a split -- is read off these fields.
    """
    out, log = trained
    rep = json.loads((out / "final_report.json").read_text())

    assert rep["trained_on_inner_val"] is False
    assert rep["n_inner_val_items"] > 0, "no selection slice was carved out"
    assert rep["n_fit_items"] == rep["n_train_items"] - rep["n_inner_val_items"], (
        "the gradient set is not the complement of the selection slice: "
        f"{rep['n_fit_items']} fitted of {rep['n_train_items']} training items "
        f"with {rep['n_inner_val_items']} held out for selection")
    assert f"gradient items: {rep['n_fit_items']} of" in log
    assert "inner-val held out of the gradient" in log

    # every control is fitted on exactly those items, or the table compares
    # models that saw different data
    for name, m in rep["baselines"].items():
        assert m["fit_items"] == rep["n_fit_items"], \
            f"{name} was fitted on {m['fit_items']} items, the head on {rep['n_fit_items']}"
    assert str(rep["n_fit_items"]) in rep["footing"]


def test_the_old_fit_rule_is_reachable_and_labels_itself(synth):
    """--train_on_inner_val reproduces the pre-fix behaviour so its effect can be
    measured, and must be impossible to mistake for the default (cf. --no_mask,
    --hvg_from all)."""
    out = synth / "p3_e16_old"
    log = _run(["src/maprna_p3/train_p3.py",
                "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
                "--esm_table", str(synth / "esm.pt"), "--out_dir", str(out),
                "--epochs", "2", "--n_hvg", "40", "--batch_size", "8",
                "--split_by", "target_gene", "--train_on_inner_val"])
    rep = json.loads((out / "final_report.json").read_text())
    assert rep["trained_on_inner_val"] is True
    assert rep["n_fit_items"] == rep["n_train_items"], \
        "the old rule fits every training item, selection slice included"
    assert "ERRATA E16" in log, "the old rule must announce itself"
    assert "E16" in rep["footing"]


def test_the_report_says_whether_the_budget_ended_the_run(trained):
    """RESULTS_REAL.md had to say this in prose three times (78/80, 150/150): a
    selected epoch equal to the budget means the model was still improving, so
    the number is a floor under that budget rather than a converged estimate."""
    out, _ = trained
    rep = json.loads((out / "final_report.json").read_text())
    assert rep["epochs_budget"] == 4
    assert rep["selection_hit_budget"] == (rep["selected_epoch"] >= 4)


def test_a_rerun_marks_the_log_boundary_instead_of_silently_concatenating(synth):
    """Found while resuming a killed run: train_log.jsonl is append-only, so a
    re-run into an existing out_dir left a log reading epochs 1..74 then 1..80
    with nothing to split on. ERRATA cites this file as evidence, so an ambiguous
    boundary is a traceability problem, not a cosmetic one.
    """
    out = synth / "rerun_boundary"
    common = ["src/maprna_p3/train_p3.py",
              "--data_dirs", str(synth / "ds1.h5ad"), str(synth / "ds2.h5ad"),
              "--esm_table", str(synth / "esm.pt"), "--out_dir", str(out),
              "--n_hvg", "40", "--batch_size", "8", "--split_by", "target_gene"]
    _run(common + ["--epochs", "2"])
    log = out / "train_log.jsonl"
    first = [json.loads(l) for l in log.read_text().splitlines()]
    starts = [r for r in first if "run_start" in r]
    assert len(starts) == 1, "the first run must open with a boundary record"
    assert starts[0]["prior_records_in_this_file"] == 0
    assert "provenance" in starts[0], "the boundary must carry the run's provenance"

    log_second = _run(common + ["--epochs", "2"])
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    starts = [i for i, r in enumerate(recs) if "run_start" in r]
    assert len(starts) == 2, "the second run must add its own boundary"
    assert recs[starts[1]]["prior_records_in_this_file"] > 0
    assert "already held" in log_second, "the re-run must say the file was not empty"

    # the last run's records are unambiguously recoverable
    tail = [r for r in recs[starts[1] + 1:] if "epoch" in r]
    assert [r["epoch"] for r in tail] == [1, 2], \
        "records after the last boundary must be exactly the latest run's epochs"
