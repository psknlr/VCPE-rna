"""Tests for the comparison assembler.

Its only real job is refusing. Almost every withdrawn number in docs/ERRATA.md was
withdrawn because two numbers were placed side by side that were not measuring the
same thing -- a pooled correlation against a within-perturbation one (E4), an epoch
chosen on the reported slice against one that was not (E3), a feature tensor
against a label (E1). None of that is visible in a finished table, so it is
checked here instead.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "assemble_comparison.py"


def prov(**args):
    return {"provenance": {"args": args, "git": {"dirty": False}}}


def seeds_report(path, value, seeds=(0, 1, 2), **over):
    a = dict(data_dirs=["/d/a.h5ad"], n_hvg=2000, hvg_from="train",
             split_by="target_gene", test_frac=0.15, min_cells=1, use_mask=True,
             esm_table="/t/esm.pt", seeds=list(seeds))
    a.update(over)
    rep = dict(pearson_dev_estimator=
                   "mean_over_perturbations_of_within_perturbation_r",
               model=dict(mean=value, n=len(seeds), sd=0.01, min=value - .01,
                          max=value + .01),
               baselines=dict(knn_esm2=dict(mean=0.20, n=len(seeds), sd=0.02,
                                            min=.18, max=.22)),
               # key name copied from run_seeds.summarise/paired output, not invented:
               # a fixture that spells it differently would hide a real mismatch
               paired_vs_baselines=dict(knn_esm2=dict(mean_difference=value - 0.20,
                                                      verdict="loses on every seed")),
               **prov(**a))
    Path(path).write_text(json.dumps(rep))


def gears_report(path, value, **over):
    a = dict(data_dirs=["/d/a.h5ad"], n_hvg=2000, hvg_from="train",
             split_by="target_gene", test_frac=0.15, min_cells=1, use_mask=True,
             esm_table="/t/esm.pt", seed=0)
    a.update(over)
    # the REAL GEARS schema: per-run list + across-runs with metric-prefixed keys,
    # and no top-level `gears` block. A fixture in any other shape would not have
    # caught the mismatch that these tests exist for.
    rep = dict(
        gears_per_run=[dict(run=i + 1, pearson_dev=value,
                            pearson_dev_estimator=
                            "mean_over_perturbations_of_within_perturbation_r")
                       for i in range(5)],
        gears_across_runs=dict(n_runs=5, pearson_dev_mean=value, pearson_dev_sd=.03,
                               pearson_dev_min=.01, pearson_dev_max=.12),
        fairness=dict(not_controlled=["tuning effort differs"]), **prov(**a))
    Path(path).write_text(json.dumps(rep))


def run(*a):
    return subprocess.run([sys.executable, str(SCRIPT), *a], capture_output=True,
                          text=True, timeout=300)


# --------------------------------------------------------------------------
# the refusals
# --------------------------------------------------------------------------

def test_it_refuses_rows_that_used_different_gene_panels(tmp_path):
    seeds_report(tmp_path / "s.json", 0.10, seeds=(0,))
    gears_report(tmp_path / "g.json", 0.15, n_hvg=500)
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode == 2
    assert "n_hvg" in p.stderr and "not measuring the same thing" in p.stderr
    assert not (tmp_path / "o.json").exists(), "it wrote a table it had refused"


def test_it_refuses_rows_that_used_different_splits(tmp_path):
    seeds_report(tmp_path / "s.json", 0.10, seeds=(0,))
    gears_report(tmp_path / "g.json", 0.15, split_by="pert")
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode == 2 and "split_by" in p.stderr


def test_it_refuses_rows_whose_seed_differs_because_the_split_depends_on_it(tmp_path):
    seeds_report(tmp_path / "s.json", 0.10, seeds=(0, 1, 2))
    gears_report(tmp_path / "g.json", 0.15, seed=7)
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode == 2 and "seed" in p.stderr


def test_it_refuses_two_different_estimators_in_one_table(tmp_path):
    """E4, machine-checked."""
    a = dict(data_dirs=["/d/a.h5ad"], n_hvg=2000, hvg_from="train",
             split_by="target_gene", test_frac=0.15, min_cells=1, use_mask=True,
             esm_table="/t/esm.pt", seed=0)
    (tmp_path / "a.json").write_text(json.dumps(dict(
        gears=dict(pearson_dev=0.1, pearson_dev_estimator="pooled"), **prov(**a))))
    (tmp_path / "b.json").write_text(json.dumps(dict(
        cpa=dict(pearson_dev=0.2,
                 pearson_dev_estimator="mean_over_perturbations_of_within_perturbation_r"),
        **prov(**a))))
    p = run("--reports", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode != 0
    assert "estimator" in p.stderr and "E4" in p.stderr


def test_an_unrecognised_report_is_an_error_not_a_silently_missing_row(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps(dict(something_else=1, **prov())))
    p = run("--reports", str(tmp_path / "x.json"), "--out_json", str(tmp_path / "o.json"))
    assert p.returncode != 0 and "cannot tell which model" in p.stderr


# --------------------------------------------------------------------------
# what it emits when the rows do agree
# --------------------------------------------------------------------------

def test_a_matching_set_produces_a_table_carrying_the_protocol_and_the_caveats(tmp_path):
    seeds_report(tmp_path / "s.json", 0.05, seeds=(0,))
    gears_report(tmp_path / "g.json", 0.15)
    out = tmp_path / "o.json"
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads(out.read_text())
    labels = [r["label"] for r in rep["rows"]]
    assert "GEARS" in labels and "P3 head" in labels and "knn_esm2" in labels
    # sorted best first, so the reader cannot miss the home model losing. Here the
    # retrieval control (0.20) beats GEARS (0.15) beats the P3 head (0.05), and the
    # table must say so in that order.
    assert labels == ["knn_esm2", "GEARS", "P3 head"], labels
    assert rep["shared_protocol"]["n_hvg"] == 2000
    assert any("tuning effort" in c for c in rep["caveats"])
    md = out.with_suffix(".md").read_text()
    assert "What is not controlled" in md
    assert "Paired per-seed comparison" in md and "loses on every seed" in md
    assert "provenance" in rep


def test_an_allowed_mismatch_is_recorded_as_a_caveat_not_forgiven(tmp_path):
    seeds_report(tmp_path / "s.json", 0.05, seeds=(0,))
    gears_report(tmp_path / "g.json", 0.15, n_hvg=500)
    out = tmp_path / "o.json"
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(out), "--allow", "n_hvg")
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads(out.read_text())
    assert any("PROTOCOL MISMATCH ALLOWED" in c and "n_hvg" in c
               for c in rep["caveats"])
    assert "n_hvg" not in rep["shared_protocol"], \
           "a field that differs must not be presented as shared"


def test_a_missing_report_path_is_an_error(tmp_path):
    p = run("--reports", str(tmp_path / "nope_*.json"), "--out_json",
            str(tmp_path / "o.json"))
    assert p.returncode != 0 and "no report matched" in p.stderr


def test_the_paired_column_reads_the_key_run_seeds_actually_writes(tmp_path):
    """Regression: the assembler read "mean" where run_seeds writes
    "mean_difference", so the paired column printed n/a with no error -- the same
    shape as the committed-JSON mismatches in ERRATA E11."""
    src = (ROOT / "src" / "maprna_p3" / "run_seeds.py").read_text()
    assert "mean_difference=float(d.mean())" in src, \
        "run_seeds no longer writes mean_difference; update the assembler with it"
    seeds_report(tmp_path / "s.json", 0.05, seeds=(0,))
    out = tmp_path / "o.json"
    p = run("--reports", str(tmp_path / "s.json"), "--out_json", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    rep = json.loads(out.read_text())
    knn = [r for r in rep["rows"] if r["label"] == "knn_esm2"][0]
    assert knn["paired_diff"] is not None, "the paired difference was dropped"
    assert abs(knn["paired_diff"] - (0.05 - 0.20)) < 1e-9
    assert "n/a" not in out.with_suffix(".md").read_text().split("Paired")[1]


def test_it_refuses_a_row_that_does_not_say_whether_it_masked(tmp_path):
    """Masked and unmasked scoring are different measurements -- ERRATA E6 -- so a
    report that does not record which it did cannot share a table with one that
    does. The unmasked configuration scores a constant block that carries no
    perturbation information, which is exactly what makes the two incomparable."""
    gears_report(tmp_path / "g.json", 0.15)
    rep = json.loads((tmp_path / "g.json").read_text())
    del rep["provenance"]["args"]["use_mask"]
    (tmp_path / "g.json").write_text(json.dumps(rep))
    p = run("--reports", str(tmp_path / "g.json"), "--out_json", str(tmp_path / "o.json"))
    assert p.returncode != 0
    assert "use_mask" in (p.stdout + p.stderr) and "E6" in (p.stdout + p.stderr)


def test_it_refuses_to_put_a_masked_and_an_unmasked_row_in_one_table(tmp_path):
    seeds_report(tmp_path / "s.json", 0.05, seeds=(0,))
    gears_report(tmp_path / "g.json", 0.15, use_mask=False)
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode == 2 and "use_mask" in p.stderr


def test_it_refuses_rows_built_from_different_embedding_tables(tmp_path):
    """The head and both ESM2 controls read the same table, so the table does not
    bias the comparison between them -- but it moves all three unequally. The
    controls are pure functions of the embedding while the head also sees the
    control profile, so a weaker table handicaps the controls more. Mixing tables
    would flatter the head invisibly."""
    seeds_report(tmp_path / "s.json", 0.05, seeds=(0,))
    gears_report(tmp_path / "g.json", 0.15, esm_table="/t/esm_150M.pt")
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode == 2 and "esm_table" in p.stderr


def test_a_multi_seed_mean_cannot_hide_a_different_estimator(tmp_path):
    """The multi-seed summary used to carry no estimator name, so it could sit
    beside a pooled-correlation row undetected -- E4 with extra steps."""
    seeds_report(tmp_path / "s.json", 0.05, seeds=(0,))
    rep = json.loads((tmp_path / "s.json").read_text())
    rep["pearson_dev_estimator"] = "pooled"
    (tmp_path / "s.json").write_text(json.dumps(rep))
    gears_report(tmp_path / "g.json", 0.15)
    p = run("--reports", str(tmp_path / "s.json"), str(tmp_path / "g.json"),
            "--out_json", str(tmp_path / "o.json"))
    assert p.returncode != 0
    assert "estimator" in p.stderr and "E4" in p.stderr


# --------------------------------------------------------------------------
# the real external-adapter schemas -- these must round-trip, because the
# assembler first refused a real GEARS report: its reader had been written to a
# schema the adapter does not use (ERRATA E11 shape, again). The fixtures below
# copy the exact structure the adapters write, not one invented to match a reader.
# --------------------------------------------------------------------------

def _ext_protocol():
    return dict(data_dirs=["/d/tian.h5ad"], n_hvg=500, hvg_from="train",
                split_by="target_gene", test_frac=0.15, min_cells=1, seed=0,
                use_mask=True, esm_table="/t/esm.pt")


def test_the_real_gears_report_schema_round_trips(tmp_path):
    """GEARS writes gears_per_run + gears_across_runs with metric-prefixed keys,
    and no top-level `gears` block. This is the exact shape that was refused."""
    rep = dict(
        gears_per_run=[
            dict(run=1, pearson_dev=-0.047,
                 pearson_dev_estimator="mean_over_perturbations_of_within_perturbation_r"),
            dict(run=2, pearson_dev=+0.017,
                 pearson_dev_estimator="mean_over_perturbations_of_within_perturbation_r"),
            dict(run=3, pearson_dev=+0.016,
                 pearson_dev_estimator="mean_over_perturbations_of_within_perturbation_r"),
        ],
        gears_across_runs=dict(n_runs=3, pearson_dev_mean=-0.0049,
                               pearson_dev_sd=0.0364, pearson_dev_min=-0.047,
                               pearson_dev_max=0.017),
        fairness=dict(not_controlled=["granularity differs on purpose"]),
        provenance=dict(args=_ext_protocol(), git=dict(dirty=False)))
    (tmp_path / "g.json").write_text(json.dumps(rep))
    out = tmp_path / "o.json"
    p = run("--reports", str(tmp_path / "g.json"), "--out_json", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    row = [r for r in json.loads(out.read_text())["rows"] if r["label"] == "GEARS"][0]
    assert abs(row["value"] - (-0.0049)) < 1e-6, "GEARS mean was not read"
    assert row["n"] == 3 and abs(row["sd"] - 0.0364) < 1e-6
    assert row["estimator"], "the estimator was dropped, so E4 could not be caught"


def test_the_real_cpa_report_schema_round_trips(tmp_path):
    """CPA writes a single `cpa` block with the metric directly and no _across_runs."""
    rep = dict(
        cpa=dict(pearson_dev=0.018, pearson_dev_pooled=0.121,
                 pearson_dev_estimator="mean_over_perturbations_of_within_perturbation_r"),
        protocol=_ext_protocol(),
        fairness=dict(not_controlled=["granularity differs on purpose"]),
        provenance=dict(args=dict(work="x"), git=dict(dirty=False)))
    (tmp_path / "c.json").write_text(json.dumps(rep))
    out = tmp_path / "o.json"
    p = run("--reports", str(tmp_path / "c.json"), "--out_json", str(out))
    assert p.returncode == 0, p.stdout + p.stderr
    row = [r for r in json.loads(out.read_text())["rows"] if r["label"] == "CPA"][0]
    assert abs(row["value"] - 0.018) < 1e-6 and row["n"] == 1


def test_the_committed_adapters_still_write_the_schema_the_assembler_reads(tmp_path):
    """A structural check on the adapter source, so a rename there fails here.

    The refusal that motivated these tests was a silent key mismatch: the adapter
    wrote one schema and the assembler read another, and nothing connected them.
    """
    gears = (ROOT / "src" / "maprna_p3" / "baseline_gears.py").read_text()
    assert "gears_across_runs" in gears and "gears_per_run" in gears
    assert "pearson_dev_mean" in gears, \
        "GEARS no longer writes pearson_dev_mean; update the assembler reader"
    cpa = (ROOT / "src" / "maprna_p3" / "baseline_cpa.py").read_text()
    assert 'cpa=score(' in cpa or '"cpa"' in cpa or "cpa=dict" in cpa
