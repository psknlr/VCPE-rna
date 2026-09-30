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
             seeds=list(seeds))
    a.update(over)
    rep = dict(model=dict(mean=value, n=len(seeds), sd=0.01, min=value - .01,
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
             seed=0)
    a.update(over)
    rep = dict(gears=dict(pearson_dev=value), gears_across_runs=dict(n_runs=5, sd=.03,
                                                                    min=.01, max=.12),
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
             seed=0)
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
