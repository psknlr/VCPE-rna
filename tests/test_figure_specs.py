"""The display-item audits must catch what they claim to catch.

Every figure in figures/ is accepted on the strength of an empty `warnings` list.
That is only as good as the checks behind it, so each check is exercised here on
a figure built to violate it. Two of these violations were found in this
repository's own figures by an independent audit before the checks existed
(coloured text, and a second sans-serif face arriving through mathtext).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "figures"))
sys.path.insert(0, str(ROOT / "tools"))

import ieee_style as S  # noqa: E402
import nature_style as ns  # noqa: E402


# ------------------------------------------------------------------ IEEE

def test_ieee_audit_passes_a_compliant_diagram(tmp_path):
    S.apply_ieee_style()
    S.set_scale()
    fig, ax, _ = S.ieee_canvas("single", 1.2)
    S.block(ax, 1.75, 0.6, 1.6, 0.4, "Linear 640→256", kind="train")
    r = S.save_ieee(fig, "ok", tmp_path, formats=("pdf", "png"))
    plt.close(fig)
    assert r["warnings"] == [] and r["column_class"] == "single"


def test_ieee_audit_flags_small_text_overflow_and_width(tmp_path):
    S.apply_ieee_style()
    S.set_scale()
    fig, ax, _ = S.ieee_canvas(5.0, 1.2)                 # not a column width
    S.block(ax, 1.0, 0.6, 0.30, 0.2, "a label far too long for its box")
    ax.text(3.0, 0.6, "tiny", fontsize=7)
    r = S.save_ieee(fig, "bad", tmp_path, formats=("pdf",))
    plt.close(fig)
    w = " ".join(r["warnings"])
    assert "not an IEEE column width" in w
    assert "text outside 9-10 pt" in w
    assert "overflows its box" in w


def test_block_kinds_stay_distinct_in_greyscale():
    assert S.check_greyscale(set(S.KINDS)) == []
    # text on every fill is legible
    for k in S.KINDS.values():
        assert S.contrast(S.TEXT, k["fc"]) >= S.MIN_CONTRAST


# ------------------------------------------------------------------ Nature

def test_nature_audit_flags_coloured_text():
    ns.apply_nature_style()
    fig = plt.figure(figsize=(3.5, 2))
    fig.text(0.1, 0.5, "an accent-coloured note", color=ns.C_ACCENT, fontsize=6)
    fig.text(0.1, 0.3, "a black note", color=ns.C_TEXT, fontsize=6)
    bad = ns.check_text_colours(fig)
    plt.close(fig)
    assert [b[0] for b in bad] == ["an accent-coloured note"]


def test_nature_pdf_audit_catches_a_second_face_and_small_scripts(tmp_path):
    ns.apply_nature_style()
    mpl.rcParams["mathtext.fontset"] = "dejavusans"   # the old default
    fig = plt.figure(figsize=(183 / 25.4, 2))
    fig.text(0.1, 0.5, r"held-out $r$ and $10^{-3}$", fontsize=6)
    p = tmp_path / "f.pdf"
    fig.savefig(p)
    plt.close(fig)
    w = " ".join(ns.audit_pdf(p))
    assert "DejaVu" in w
    assert "below the 5 pt floor" in w
    ns.apply_nature_style()                          # restore for later tests


def test_nature_pdf_audit_passes_the_corrected_style(tmp_path):
    ns.apply_nature_style()
    fig = plt.figure(figsize=(183 / 25.4, 2))
    fig.text(0.1, 0.5, r"held-out $r$, $k/n$ = 0.10", fontsize=6)
    p = tmp_path / "g.pdf"
    fig.savefig(p)
    plt.close(fig)
    assert ns.audit_pdf(p) == []


# ------------------------------------------------------------------ facts

def test_architecture_figure_reads_its_numbers_from_the_code():
    torch = pytest.importorskip("torch")
    import ieee_architecture as A
    sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))
    from model_dev import DeviationModel
    F = A.architecture_facts()
    m = DeviationModel(torch.zeros(4, F["d_table"]), np.arange(F["n_hvg"]),
                       n_ds=F["n_ds"], d_model=F["d_model"])
    assert F["n_params"] == sum(p.numel() for p in m.parameters())
    assert F["d_table"] == 2 * F["d_own"]
    assert F["concat_dim"] == 4 * F["d_model"] + 3 + F["ds_emb"][1]
    assert F["inner_val_held_out"] is True
    assert F["optimiser"] == "AdamW" and F["clip"] == 1.0


def test_paired_intervals_in_figures_use_the_t_quantile():
    from scipy import stats
    import figdata as F
    s = F._summary("results/real_150M/A_seeds/seed_summary.json")
    p = F.paired(s, "ridge_esm2")
    d = np.asarray(p["per_seed"])
    half = (p["hi"] - p["lo"]) / 2
    assert half == pytest.approx(stats.t.ppf(0.975, 2) * d.std(ddof=1) / np.sqrt(3))
    # the stored, 1.96-based interval is narrower and is never what is drawn
    lo0, hi0 = p["stored_normal_ci"]
    assert (hi0 - lo0) / 2 < half / 2


def test_train_log_parser_splits_a_resumed_log_without_a_boundary():
    """graph_long seed 2 was killed at epoch 74 and re-run: 74 + 80 records."""
    import figdata as F
    L = F.train_log("results/real_150M_graph_long/A_seeds/seed2/train_log.jsonl")
    assert len(L["epochs"]) == 80
    assert L["final"]["selected_epoch"] <= 80


# ------------------------------------------------------------------ simulation

def test_simulated_traps_behave_as_documented():
    import simulate_protocol_traps as T

    class A:
        n_genes, n_train, n_test, replicates, k = 200, 80, 30, 2, 20
        informative_r, seed = 0.3, 0

    rng = np.random.default_rng(0)
    ties = T.trap_ties(A, rng)
    assert np.allclose(ties["series"]["tie_aware"]["mean"], A.k / A.n_genes)
    assert ties["series"]["index_ties_sorted"]["mean"][-1] > 2 * A.k / A.n_genes
    um = T.trap_unmeasured(A, rng)
    assert np.allclose(um["series"]["masked"]["mean"], 0.0)
    assert um["series"]["unmasked"]["mean"][-1] > 0.2
    sh = T.trap_shared(A, rng)
    assert sh["series"]["no_info_raw"]["mean"][-1] > 0.8
    assert np.allclose(sh["series"]["no_info_residual"]["mean"], 0.0)
