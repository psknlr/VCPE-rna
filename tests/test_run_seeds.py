"""Tests for multi-seed aggregation and the paired model-vs-baseline comparison.

The point of pairing: model and baselines share a seed, hence a split, so the
paired difference removes the split-to-split variance that dominates the marginal
spread. On this data scale that variance is large enough to flip a verdict --
which is exactly why single-seed point estimates were never sufficient.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))
sys.path.insert(0, str(ROOT / "src"))

from run_seeds import summarise  # noqa: E402


def test_summarise_reports_mean_sd_range_and_ci():
    s = summarise([0.10, 0.20, 0.30], "x")
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(0.20)
    assert s["sd"] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
    assert (s["min"], s["max"]) == (0.10, 0.30)
    lo, hi = s["ci95_mean"]
    assert lo < s["mean"] < hi


def test_summarise_uses_the_sample_sd_not_the_population_sd():
    """With n=3 the difference is ~22%, and the population form understates it."""
    vals = [0.1, 0.2, 0.3]
    s = summarise(vals, "x")
    assert s["sd"] == pytest.approx(np.std(vals, ddof=1))
    assert s["sd"] != pytest.approx(np.std(vals))


def test_summarise_drops_none_and_nan():
    s = summarise([0.1, None, float("nan"), 0.3], "x")
    assert s["n"] == 2
    assert s["mean"] == pytest.approx(0.2)


def test_summarise_handles_a_single_seed_without_inventing_a_spread():
    s = summarise([0.42], "x")
    assert s["n"] == 1 and s["sd"] is None and s["ci95_mean"] is None


def test_summarise_of_nothing_is_empty_not_an_error():
    assert summarise([None, float("nan")], "x")["n"] == 0


# --------------------------------------------------------------------------
# the paired comparison, exercised through the script
# --------------------------------------------------------------------------

def _fake_reports(tmp_path, per_seed):
    """Write final_report.json files as train_p3 would, then aggregate them."""
    for s, (model, knn) in per_seed.items():
        d = tmp_path / f"seed{s}"
        d.mkdir(parents=True)
        (d / "final_report.json").write_text(json.dumps({
            "pearson_dev": model, "selected_epoch": 3, "split_by": "target_gene",
            "baselines": {"knn_esm2": {"pearson_dev": knn},
                          "zero": {"pearson_dev": 0.0}}}))
    out = tmp_path / "summary.json"
    p = subprocess.run(
        [sys.executable, str(ROOT / "src" / "maprna_p3" / "run_seeds.py"),
         "--seeds", *[str(s) for s in per_seed], "--out_dir", str(tmp_path),
         "--skip_existing", "--summary_json", str(out)],
        capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(out.read_text()), p.stdout


def test_consistent_win_is_reported_as_such(tmp_path):
    rep, log = _fake_reports(tmp_path, {0: (0.30, 0.10), 1: (0.28, 0.12),
                                        2: (0.33, 0.09)})
    d = rep["paired_vs_baselines"]["knn_esm2"]
    assert d["seeds_where_model_wins"] == 3
    assert "every seed" in d["verdict"]
    assert d["mean_difference"] > 0


def test_consistent_loss_is_reported_and_warned_about(tmp_path):
    rep, log = _fake_reports(tmp_path, {0: (0.05, 0.20), 1: (0.04, 0.18),
                                        2: (0.06, 0.22)})
    d = rep["paired_vs_baselines"]["knn_esm2"]
    assert d["seeds_where_model_wins"] == 0
    assert "loses" in d["verdict"]
    assert "loses on EVERY seed" in log


def test_a_mixed_result_is_not_rounded_into_a_win(tmp_path):
    """The case a single seed would misreport either way.

    Here the mean paired difference is positive, but the model loses on one of
    three seeds. Reporting only the mean would read as a win.
    """
    rep, _ = _fake_reports(tmp_path, {0: (0.20, 0.05), 1: (0.08, 0.12),
                                      2: (0.18, 0.07)})
    d = rep["paired_vs_baselines"]["knn_esm2"]
    assert d["mean_difference"] > 0
    assert d["seeds_where_model_wins"] == 2
    assert d["verdict"].startswith("mixed")


def test_paired_ci_can_exclude_zero_where_the_marginal_intervals_overlap(tmp_path):
    """Why pairing is the right test.

    Model and baseline both swing widely across seeds (their marginal intervals
    overlap heavily), but within each seed the model is consistently ahead by a
    similar margin. The paired CI sees that; comparing marginals would not.
    """
    rep, _ = _fake_reports(tmp_path, {0: (0.50, 0.44), 1: (0.20, 0.14),
                                      2: (0.35, 0.29), 3: (0.10, 0.04)})
    m, b = rep["model"], rep["baselines"]["knn_esm2"]
    assert m["ci95_mean"][0] < b["ci95_mean"][1], "marginal intervals should overlap"
    lo, hi = rep["paired_vs_baselines"]["knn_esm2"]["ci95_mean_difference"]
    assert lo > 0, "the paired difference should be clearly positive"


def test_summary_records_the_split_and_the_selected_epochs(tmp_path):
    rep, _ = _fake_reports(tmp_path, {0: (0.1, 0.0), 1: (0.2, 0.0)})
    assert rep["split_by"] == ["target_gene"]
    assert rep["selected_epochs"] == [3, 3]
    assert rep["n_seeds"] == 2
    assert "provenance" in rep


def test_forwarding_a_banned_argument_is_refused():
    """--seed and --out_dir are set per run; silently ignoring them would make
    every seed write to the same place."""
    p = subprocess.run(
        [sys.executable, str(ROOT / "src" / "maprna_p3" / "run_seeds.py"),
         "--out_dir", "/tmp/x", "--", "--seed", "3"],
        capture_output=True, text=True, timeout=120)
    assert p.returncode != 0
    assert "set per run" in (p.stdout + p.stderr)
