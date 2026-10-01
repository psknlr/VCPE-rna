"""Forward-pass tests for both model heads.

These need torch and are skipped without it, but no data assets: every tensor is
synthetic. They pin the architectural fixes from docs/ERRATA.md that a purely
numerical metric test cannot reach -- in particular that the ASO encoder is no
longer permutation-invariant (E9), that the frozen ESM2 table no longer inflates
checkpoints (E12f), and that each conditioning axis of the response head can
actually be ablated on its own (the per-axis ablation the P2.3-B report deferred).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "efficacy"))
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))
sys.path.insert(0, str(ROOT / "src" / "maprna_p1"))


# --------------------------------------------------------------------------
# ASO efficacy head
# --------------------------------------------------------------------------

def _aso_batch(L=20, seed=0):
    from model_efficacy import BASES, BACKBONES, SUGARS
    rng = np.random.default_rng(seed)
    base = torch.tensor([[BASES[c] for c in rng.choice(list("ACGT"), L)]])
    # a 5-10-5-style gapmer: modified wings, DNA gap
    sug = torch.tensor([[SUGARS["MOE"]] * 5 + [SUGARS["DNA"]] * (L - 10)
                        + [SUGARS["MOE"]] * 5])
    bb = torch.full((1, L), BACKBONES["PS"], dtype=torch.long)
    pad = torch.zeros(1, L, dtype=torch.bool)
    return base, sug, bb, pad


def _aso_model(use_pos_emb=True, d_esm=32, n_genes=8):
    from model_efficacy import EfficacyHead
    torch.manual_seed(0)
    esm = torch.randn(n_genes, d_esm)
    return EfficacyHead(esm, n_cell_lines=4, use_pos_emb=use_pos_emb)


def _aso_run(m, base, sug, bb, pad):
    with torch.no_grad():
        return m(base, sug, bb, pad, torch.tensor([0]), torch.tensor([0]),
                 torch.zeros(1), torch.ones(1)).item()


def test_aso_head_runs():
    m = _aso_model()
    assert np.isfinite(_aso_run(m, *_aso_batch()))


def test_positional_encoding_makes_order_matter():
    """Reversing the molecule must change the prediction.

    Without a positional embedding the encoder is permutation-equivariant and
    the pooling permutation-invariant, so the whole sequence branch is a bag of
    (base, sugar, backbone) triples -- exactly the defect in ERRATA E9.
    """
    m = _aso_model(use_pos_emb=True).eval()
    base, sug, bb, pad = _aso_batch()
    a = _aso_run(m, base, sug, bb, pad)
    b = _aso_run(m, base.flip(1), sug.flip(1), bb.flip(1), pad)
    assert abs(a - b) > 1e-5, "order-sensitive model gave identical predictions"


def test_without_positional_encoding_order_is_invisible():
    """The legacy configuration: same multiset of triples -> same prediction.

    This is what every pre-v4 checkpoint does, and why the "~11% plain DNA vs
    ~38% LNA/cEt wings" showcase could only have reflected modification COUNT.
    """
    m = _aso_model(use_pos_emb=False).eval()
    base, sug, bb, pad = _aso_batch()
    a = _aso_run(m, base, sug, bb, pad)
    b = _aso_run(m, base.flip(1), sug.flip(1), bb.flip(1), pad)
    assert abs(a - b) < 1e-4, (
        "a bag-of-triples encoder should be invariant to reversal; got "
        f"{a:.6f} vs {b:.6f}")


def test_moving_modifications_changes_a_position_aware_model():
    """Same NUMBER of modified sugars, different placement."""
    from model_efficacy import SUGARS
    m = _aso_model(use_pos_emb=True).eval()
    base, sug, bb, pad = _aso_batch()
    L = sug.shape[1]
    middle = torch.tensor([[SUGARS["DNA"]] * 5 + [SUGARS["MOE"]] * 10
                           + [SUGARS["DNA"]] * (L - 15)])
    assert middle.eq(SUGARS["MOE"]).sum() == sug.eq(SUGARS["MOE"]).sum()
    a = _aso_run(m, base, sug, bb, pad)
    b = _aso_run(m, base, middle, bb, pad)
    assert abs(a - b) > 1e-5


def test_esm_table_is_not_in_the_state_dict():
    """ERRATA E12f: persisting the frozen public table added ~400MB per ckpt."""
    m = _aso_model()
    assert "esm_table" not in m.state_dict()


def test_pad_index_does_not_freeze_a_real_chemistry_class():
    """ERRATA E9: pre-v4 padding_idx=0 pinned the DNA sugar and PO backbone rows.

    Those are the gap chemistry of every gapmer and the unmodified backbone, so
    they must be trainable; only the dedicated PAD row may be frozen at zero.
    """
    from model_efficacy import BACKBONES, PAD, SUGARS
    m = _aso_model()
    assert m.sugar_emb.padding_idx == PAD
    assert m.bb_emb.padding_idx == PAD
    assert SUGARS["DNA"] != PAD and BACKBONES["PO"] != PAD
    assert torch.allclose(m.sugar_emb.weight[PAD], torch.zeros(16))
    assert not torch.allclose(m.sugar_emb.weight[SUGARS["DNA"]], torch.zeros(16))


def test_arch_config_round_trips_through_a_checkpoint():
    from model_efficacy import load_efficacy_head
    m = _aso_model(use_pos_emb=True)
    ck = {"model_state_dict": m.state_dict(), "arch_config": m.arch_config()}
    m2, cfg = load_efficacy_head(ck, m.esm_table, 4)
    assert cfg["use_pos_emb"] is True
    assert m2.pos_emb is not None


def _legacy_aso_model(d_esm=32, n_genes=8):
    """A genuine pre-v4 model: no positional embedding, legacy vocabularies."""
    from model_efficacy import EfficacyHead
    torch.manual_seed(0)
    esm = torch.randn(n_genes, d_esm)
    return EfficacyHead(esm, n_cell_lines=4, use_pos_emb=False, legacy_vocab=True)


def test_pre_v4_checkpoint_is_rebuilt_as_itself():
    """A pre-v4 checkpoint must be rebuilt as itself, not reinterpreted.

    The layout is inferred from the weight shapes, so a checkpoint with no
    arch_config field still loads correctly -- and a stale field cannot override
    what the weights say.
    """
    from model_efficacy import load_efficacy_head
    legacy = _legacy_aso_model()
    sd = dict(legacy.state_dict())
    sd["esm_table"] = torch.randn(8, 32)          # pre-v4 persisted this
    m2, cfg = load_efficacy_head({"model_state_dict": sd}, legacy.esm_table, 4)
    assert cfg["use_pos_emb"] is False
    assert cfg["legacy_vocab"] is True
    assert m2.pos_emb is None
    # and it actually runs, on legacy-vocabulary inputs
    L = 12
    base = torch.zeros(1, L, dtype=torch.long)
    with torch.no_grad():
        out = m2(base, base.clone(), base.clone(), torch.zeros(1, L, dtype=torch.bool),
                 torch.tensor([0]), torch.tensor([0]), torch.zeros(1), torch.ones(1))
    assert torch.isfinite(out).all()


def test_v4_checkpoint_is_detected_from_weights_alone():
    from model_efficacy import load_efficacy_head
    m = _aso_model(use_pos_emb=True)
    _, cfg = load_efficacy_head({"model_state_dict": m.state_dict()}, m.esm_table, 4)
    assert cfg["use_pos_emb"] is True
    assert cfg["legacy_vocab"] is False


def test_weights_win_over_a_stale_arch_config():
    """A hand-edited or stale field must not change how weights are read."""
    from model_efficacy import load_efficacy_head
    m = _aso_model(use_pos_emb=True)
    ck = {"model_state_dict": m.state_dict(),
          "arch_config": {"use_pos_emb": False, "legacy_vocab": True}}
    m2, cfg = load_efficacy_head(ck, m.esm_table, 4)
    assert cfg["use_pos_emb"] is True and cfg["legacy_vocab"] is False
    assert m2.pos_emb is not None


# --------------------------------------------------------------------------
# P3 deviation head: per-axis ablation hooks
# --------------------------------------------------------------------------

def _dev_model(n_hvg=12, vocab=20, d_esm=16, n_ds=3, seed=0):
    from model_dev import DeviationModel
    torch.manual_seed(seed)
    esm = torch.randn(vocab, d_esm)
    hvg = np.arange(n_hvg)
    return DeviationModel(esm, hvg, n_ds=n_ds, d_model=8, hidden=16)


def _dev_inputs(bp=5, n_hvg=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.arange(bp),                                   # pert_rows
            torch.randint(0, 3, (bp,), generator=g),            # ds_idx
            torch.randn(bp, n_hvg, generator=g))                # ctrl_feat


def test_dev_head_runs_and_shape_is_right():
    m = _dev_model().eval()
    rows, ds, cf = _dev_inputs()
    with torch.no_grad():
        out = m(rows, ds, cf)
    assert out.shape == (5, 12)
    assert torch.isfinite(out).all()


def test_dev_esm_table_is_not_persisted():
    """ERRATA E12f on the response line: 405MB of a 428MB checkpoint."""
    assert "esm_table" not in _dev_model().state_dict()


def test_each_conditioning_axis_can_be_ablated_independently():
    """The hooks the per-axis ablation needs, which did not exist before.

    Overriding the ESM2 vector alone leaves is_target and is_neighbor carrying
    perturbation identity -- the caveat the P2.3-B report raised and never
    tested. Each override must change the output on its own.
    """
    m = _dev_model().eval()
    rows, ds, cf = _dev_inputs()
    is_nb, is_tgt = m.indicator_features(rows)
    with torch.no_grad():
        base = m(rows, ds, cf)
        esm_off = m(rows, ds, cf, pert_esm_override=torch.zeros_like(m.esm_table[rows]))
        tgt_off = m(rows, ds, cf, is_tgt_override=torch.zeros_like(is_tgt))
        nb_off = m(rows, ds, cf, is_nb_override=torch.ones_like(is_nb))
    for name, alt in (("esm", esm_off), ("is_tgt", tgt_off), ("is_nb", nb_off)):
        assert not torch.allclose(base, alt), f"{name} override had no effect"


def test_is_target_override_does_not_move_the_expression_gate():
    """The v2-1a gate is a structural prior, not a conditioning channel.

    If ablating the is_target FEATURE also relocated the gate, the ablation
    would confound two things at once -- the same mistake as ERRATA E7.
    """
    m = _dev_model().eval()
    rows, ds, cf = _dev_inputs()
    _, is_tgt = m.indicator_features(rows)
    pe = torch.zeros(len(rows))          # low expression -> gate clamps hard
    with torch.no_grad():
        out = m(rows, ds, cf, pert_ctrl_expr=pe,
                is_tgt_override=torch.zeros_like(is_tgt))
    # the gate must still be applied on the TRUE target rows
    gated = out[is_tgt]
    assert torch.abs(gated).max() < torch.abs(out).max()


def test_shuffling_the_dataset_index_changes_the_output():
    """Needed for the ds-shuffle control: dataset identity is a real input."""
    m = _dev_model().eval()
    rows, ds, cf = _dev_inputs()
    with torch.no_grad():
        a = m(rows, torch.zeros_like(ds), cf)
        b = m(rows, torch.full_like(ds, 2), cf)
    assert not torch.allclose(a, b)


def test_indicator_features_match_what_forward_builds():
    """indicator_features must not drift from the forward pass's own logic."""
    m = _dev_model().eval()
    rows, ds, cf = _dev_inputs()
    is_nb, is_tgt = m.indicator_features(rows)
    with torch.no_grad():
        a = m(rows, ds, cf)
        b = m(rows, ds, cf, is_tgt_override=is_tgt, is_nb_override=is_nb)
    assert torch.allclose(a, b, atol=1e-6)
