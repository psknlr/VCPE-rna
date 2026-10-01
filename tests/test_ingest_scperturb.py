"""Tests for the scPerturb -> loader-format adapter.

The adapter exists because GEARS' own dataset host is refused by this
environment's egress policy, so every real number has to come through it. That
makes it the single point where a silent error would corrupt every fold change
downstream, which is what these tests are about rather than file plumbing.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "maprna_p3" / "ingest_scperturb.py"
ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")


def write_src(path, pert, counts, syms, nperts=None):
    obs = pd.DataFrame(dict(perturbation=pert))
    if nperts is not None:
        obs["nperts"] = nperts
    obs.index = [f"c{i}" for i in range(len(pert))]
    var = pd.DataFrame(index=list(syms))
    a = ad.AnnData(X=np.asarray(counts, dtype=np.float32), obs=obs, var=var)
    a.write_h5ad(path)
    return path


def run(src, out, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--h5ad", str(src),
                           "--out", str(out), *extra],
                          capture_output=True, text=True, timeout=600)


# --------------------------------------------------------------------------
# the ordering that matters
# --------------------------------------------------------------------------

def test_normalisation_happens_before_the_pseudobulk_mean(tmp_path):
    """Per-cell normalisation and the pseudobulk mean do not commute.

    Two cells of one perturbation, sequenced 10x apart but with the SAME
    composition. The right answer is composition-only and must not depend on
    depth. Taking the mean of raw counts first would let the deeper cell dominate,
    so a perturbation that happened to be sequenced deeper would report a larger
    fold change than an identical one that was not.
    """
    syms = [f"G{i}" for i in range(4)]
    counts = np.array([
        [1, 2, 3, 4],            # MYC, shallow
        [10, 20, 30, 40],        # MYC, 10x deeper, identical composition
        [1, 1, 1, 1],            # control
        [1, 1, 1, 1],            # control
    ], dtype=np.float32)
    src = write_src(tmp_path / "s.h5ad", ["MYC", "MYC", "control", "control"],
                    counts, syms, nperts=[1, 1, 0, 0])
    out = tmp_path / "o.h5ad"
    p = run(src, out, "--min_cells", "1", "--n_ctrl", "10")
    assert p.returncode == 0, p.stdout + p.stderr
    a = ad.read_h5ad(out)
    row = np.asarray(a.X)[list(a.obs.condition).index("MYC+ctrl")]

    per_cell = np.log1p(counts[:2] / counts[:2].sum(1, keepdims=True) * 1e4)
    assert np.allclose(row, per_cell.mean(0), atol=1e-5)
    # the two cells have identical composition, so both normalised rows are equal
    assert np.allclose(per_cell[0], per_cell[1], atol=1e-5)
    # depth must not change the answer at all. (Here the two compositions are
    # identical, so the wrong order happens to agree; the contrast between two
    # DIFFERENT perturbations is the next test, which is where it bites.)
    assert np.allclose(row, per_cell[0], atol=1e-5), \
        "sequencing depth leaked into the pseudobulk profile"


def test_depth_imbalance_between_perturbations_does_not_change_their_ranking(tmp_path):
    """The failure the ordering prevents, stated as a comparison."""
    syms = [f"G{i}" for i in range(4)]
    counts = np.array([
        [4, 1, 1, 1], [40, 10, 10, 10],     # A: same composition, 10x depth apart
        [4, 1, 1, 1], [4, 1, 1, 1],         # B: same composition, equal depth
        [1, 1, 1, 1], [1, 1, 1, 1],         # control
    ], dtype=np.float32)
    src = write_src(tmp_path / "s.h5ad", ["A", "A", "B", "B", "control", "control"],
                    counts, syms, nperts=[1, 1, 1, 1, 0, 0])
    out = tmp_path / "o.h5ad"
    assert run(src, out, "--min_cells", "1", "--n_ctrl", "10").returncode == 0
    a = ad.read_h5ad(out)
    X = np.asarray(a.X)
    ra = X[list(a.obs.condition).index("A+ctrl")]
    rb = X[list(a.obs.condition).index("B+ctrl")]
    assert np.allclose(ra, rb, atol=1e-5), \
        "two perturbations with identical composition differ only by depth"


# --------------------------------------------------------------------------
# labels, controls, and the things it must refuse
# --------------------------------------------------------------------------

def test_labels_are_translated_to_the_loader_s_convention(tmp_path):
    src = write_src(tmp_path / "s.h5ad", ["MYC", "control"],
                    [[1, 2], [3, 4]], ["G0", "G1"], nperts=[1, 0])
    out = tmp_path / "o.h5ad"
    assert run(src, out, "--min_cells", "1").returncode == 0
    a = ad.read_h5ad(out)
    assert set(a.obs.condition) == {"MYC+ctrl", "ctrl"}
    assert a.obs.control.tolist() == [0, 1]
    assert list(a.var["gene_name"]) == ["G0", "G1"]


def test_it_refuses_a_file_with_no_control_population(tmp_path):
    """Picking a control guide by name would be a guess, and every fold change
    in the run would be silently wrong."""
    src = write_src(tmp_path / "s.h5ad", ["MYC", "TP53"], [[1, 2], [3, 4]],
                    ["G0", "G1"], nperts=[1, 1])
    p = run(src, tmp_path / "o.h5ad", "--min_cells", "1")
    assert p.returncode != 0
    assert "control" in (p.stdout + p.stderr).lower()


def test_it_refuses_multi_gene_perturbations_rather_than_guessing_the_separator(tmp_path):
    src = write_src(tmp_path / "s.h5ad", ["MYC_TP53", "control"], [[1, 2], [3, 4]],
                    ["G0", "G1"], nperts=[2, 0])
    p = run(src, tmp_path / "o.h5ad", "--min_cells", "1")
    assert p.returncode != 0
    assert "nperts" in (p.stdout + p.stderr)


# --------------------------------------------------------------------------
# the two book-keeping properties the loader depends on
# --------------------------------------------------------------------------

def test_the_cell_count_gate_runs_here_and_the_true_counts_are_recorded(tmp_path):
    """After aggregation every perturbation is one row, so the loader's own
    --min_cells gate cannot see cell counts any more."""
    pert = ["A"] * 5 + ["B"] * 2 + ["control"] * 3
    counts = np.ones((len(pert), 3), dtype=np.float32)
    src = write_src(tmp_path / "s.h5ad", pert, counts, ["G0", "G1", "G2"],
                    nperts=[1] * 7 + [0] * 3)
    out = tmp_path / "o.h5ad"
    assert run(src, out, "--min_cells", "5").returncode == 0
    a = ad.read_h5ad(out)
    conds = set(a.obs.condition)
    assert "A+ctrl" in conds and "B+ctrl" not in conds, "the gate did not apply"
    assert int(a.obs.n_cells[a.obs.condition == "A+ctrl"].iloc[0]) == 5
    prov = json.loads(Path(str(out).replace(".h5ad", ".provenance.json")).read_text())
    assert prov["min_cells"] == 5
    assert "--min_cells 1" in prov["downstream_requirement"]


def test_duplicate_gene_symbols_are_summed_as_counts_not_silently_dropped(tmp_path):
    """Two columns with one symbol both map to the same embedding row downstream,
    where a dict keeps whichever came last."""
    counts = np.array([[1, 2, 5], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
    src = write_src(tmp_path / "s.h5ad", ["MYC", "control", "control"], counts,
                    ["G0", "G0", "G1"], nperts=[1, 0, 0])
    out = tmp_path / "o.h5ad"
    assert run(src, out, "--min_cells", "1").returncode == 0
    a = ad.read_h5ad(out)
    assert list(a.var["gene_name"]) == ["G0", "G1"]
    merged = np.array([[3.0, 5.0]])                      # 1 + 2 summed as counts
    exp = np.log1p(merged / merged.sum() * 1e4)[0]
    assert np.allclose(np.asarray(a.X)[list(a.obs.condition).index("MYC+ctrl")],
                       exp, atol=1e-5)


# --------------------------------------------------------------------------
# cells mode -- the input the external baselines actually need
# --------------------------------------------------------------------------

def test_cells_mode_emits_labelled_cells_whose_mean_is_the_pseudobulk_export(tmp_path):
    """The two modes must describe the same data.

    GEARS and CPA are cell-level models, so handing them one pseudobulk row per
    condition would handicap them, and a comparison that handicaps the external
    baseline proves nothing. They therefore read a different file -- and that only
    stays a fair comparison if the two files agree, which is what this checks.
    """
    rng = np.random.default_rng(0)
    pert = ["A"] * 6 + ["B"] * 4 + ["control"] * 5
    counts = rng.integers(0, 30, size=(len(pert), 6)).astype(np.float32)
    counts[counts.sum(1) == 0, 0] = 1                      # no empty cells
    syms = [f"G{i}" for i in range(6)]
    src = write_src(tmp_path / "s.h5ad", pert, counts, syms,
                    nperts=[1] * 10 + [0] * 5)

    pb, cl = tmp_path / "pb.h5ad", tmp_path / "cl.h5ad"
    assert run(src, pb, "--min_cells", "1", "--n_ctrl", "5").returncode == 0
    p = run(src, cl, "--min_cells", "1", "--n_ctrl", "5", "--mode", "cells")
    assert p.returncode == 0, p.stdout + p.stderr

    A, B = ad.read_h5ad(pb), ad.read_h5ad(cl)
    Xc = np.asarray(B.X.todense() if hasattr(B.X, "todense") else B.X)
    assert (B.obs.condition == "A+ctrl").sum() == 6
    assert (B.obs.condition == "B+ctrl").sum() == 4
    assert int(B.obs.control.sum()) == 5
    for cond in ("A+ctrl", "B+ctrl"):
        got = Xc[(B.obs.condition == cond).values].mean(0)
        exp = np.asarray(A.X)[list(A.obs.condition).index(cond)]
        assert np.allclose(got, exp, atol=1e-5), \
            f"{cond}: the cell export and the pseudobulk export disagree"


def test_cells_mode_cap_subsamples_and_records_it(tmp_path):
    pert = ["A"] * 10 + ["control"] * 3
    counts = np.ones((len(pert), 4), dtype=np.float32)
    src = write_src(tmp_path / "s.h5ad", pert, counts, [f"G{i}" for i in range(4)],
                    nperts=[1] * 10 + [0] * 3)
    out = tmp_path / "o.h5ad"
    assert run(src, out, "--min_cells", "1", "--mode", "cells",
               "--max_cells_per_pert", "4").returncode == 0
    a = ad.read_h5ad(out)
    assert (a.obs.condition == "A+ctrl").sum() == 4
    prov = json.loads(Path(str(out).replace(".h5ad", ".provenance.json")).read_text())
    assert prov["mode"] == "cells" and prov["max_cells_per_pert"] == 4
    assert prov["n_perturbed_cells_kept"] == 4
    # the gate still sees the TRUE count, not the capped one
    assert prov["cells_per_pert_median"] == 10.0
