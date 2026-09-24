#!/usr/bin/env python
"""ASO efficacy inference: sequence + target gene (+ cell line) -> knockdown depth.

Input : DNA sequence, 10-30 nt, non-ACGT characters stripped. Chemistry defaults
        to uniform PS backbone with UNMODIFIED (DNA) wings -- which is a
        PS-DNA oligonucleotide, not a gapmer. Pass `wing_mod` explicitly
        ('moe' / 'lna' -> cEt / 'f') to place the molecule inside the training
        distribution (~53% 5-10-5 MOE, ~30% 3-10-3 cEt).
Dose  : not exposed in the public API; every call is made at the training-mean
        dose with the missing flag set, mirroring how missing dose was handled
        during training.
Cells : exact cell2id match (case-insensitive); unknown -> mean of the trained
        cell embeddings. That mean vector never occurred during training and its
        behaviour has not been validated.
Output: inhibition_pct (clipped to 0-95) and kd = inhibition_pct / 100, which
        are guaranteed to agree with one another.

Accuracy: this module quotes whatever held-out metrics the loaded checkpoint
carries and refuses to invent one. Checkpoints written before v4 were selected
on the same slice they reported, and their sequence encoder has no positional
embedding; `position_aware` in the output says which generation is loaded.

Scope: the model takes sequence + wing chemistry + cell line. Delivery
(GalNAc / LNP / route / tissue exposure) is NOT modelled, and outputs are
intrinsic in-vitro potency, not tissue-level efficacy.

Usage:
  from predict_service import predict_inhibition
  predict_inhibition("TGCATCGTACGTAGCTGATC", "APOC3", cell_line="hepg2",
                     wing_mod="moe")
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.abspath(os.path.join(
    HERE, "..", "..", "results", "efficacy_v1", "ckpt_efficacy_v1.pt"))
DEFAULT_ESM = os.path.abspath(os.path.join(
    HERE, "..", "..", "data", "drive_weights",
    "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"))

_pack_cache = {}


def _load(ckpt_fp=None, esm_fp=None):
    ckpt_fp = ckpt_fp or DEFAULT_CKPT
    esm_fp = esm_fp or DEFAULT_ESM
    key = (ckpt_fp, esm_fp)
    if key in _pack_cache:
        return _pack_cache[key]
    from model_efficacy import load_efficacy_head, vocabs
    ck = torch.load(ckpt_fp, map_location="cpu", weights_only=False)
    tab = torch.load(esm_fp, map_location="cpu", weights_only=False)
    symbols = list(tab.keys())
    esm = torch.stack([v.float() for v in tab.values()])
    gene2row = {s: i for i, s in enumerate(symbols)}
    n_cell = ck["model_state_dict"]["cell_emb.weight"].shape[0]
    # load_efficacy_head reads `arch_config` from the checkpoint, so a pre-v4
    # checkpoint is rebuilt with its own (legacy, position-free) architecture
    # and its own vocabularies instead of being silently reinterpreted under
    # the current ones.
    m, arch = load_efficacy_head(ck, esm, n_cell)
    m.eval()
    n_trained = n_cell - 1                      # 末行未训练（未知回退保留位）
    cell_mean = m.cell_emb.weight[:n_trained].mean(0).detach()
    b_voc, s_voc, k_voc = vocabs(arch.get("legacy_vocab", True))
    pack = dict(model=m, gene2row=gene2row, cell2id=ck["cell2id"],
                dose_mu=float(ck["dose_mu"]), cell_mean=cell_mean,
                arch=arch, bases=b_voc, sugars=s_voc, backbones=k_voc,
                metrics=ck.get("metrics", {}))
    _pack_cache[key] = pack
    return pack


# Vocabularies are taken from the loaded checkpoint via model_efficacy.vocabs();
# module-level index tables were removed because they silently assumed the
# pre-v4 layout, in which index 0 meant "DNA"/"PO" rather than PAD.
# 训练词表无 LNA 类：LNA 与 cEt 同属约束型 BNA 家族，映射到 cEt（模型已学的最近类）
WING_MAP = {"dna": "DNA", "unmodified": "DNA", "moe": "MOE",
            "lna": "cEt", "cet": "cEt", "f": "F", "mix": "MOE"}


def predict_inhibition(seq, target_gene, cell_line=None,
                       wing_mod=None, wing_len=3,
                       ckpt_fp=None, esm_fp=None):
    """返回 dict；error 键存在表示预测失败（调用方回退手输效率）。"""
    pack = _load(ckpt_fp, esm_fp)
    m = pack["model"]
    seq = "".join(ch for ch in str(seq).upper() if ch in "ACGT")
    if len(seq) < 10 or len(seq) > 30:
        return dict(error=f"序列长度需 10-30nt（当前 {len(seq)}）")
    row = pack["gene2row"].get(target_gene)
    if row is None:
        return dict(error=f"靶基因 {target_gene} 不在 ESM 表覆盖范围")

    L = len(seq)
    # Vocabularies come from the loaded checkpoint's architecture, not from
    # module-level constants: the v4 vocabularies reserve index 0 for PAD, so a
    # hard-coded table would map every symbol one position off for one of the
    # two checkpoint generations.
    B_VOC, S_VOC, K_VOC = pack["bases"], pack["sugars"], pack["backbones"]
    base = torch.tensor([[B_VOC.get(ch, B_VOC["N"]) for ch in seq]], dtype=torch.long)
    wing_eff = WING_MAP.get(str(wing_mod or "dna").lower(), "DNA")
    wl = max(0, min(int(wing_len), (L - 1) // 2))
    sug_idx = S_VOC[wing_eff]
    sug_row = [sug_idx if (i < wl or i >= L - wl) else S_VOC["DNA"] for i in range(L)]
    sug = torch.tensor([sug_row], dtype=torch.long)
    # Backbone defaults to uniform PS. Note that with wing_mod="dna" (the
    # default) the resulting molecule is a fully-DNA phosphorothioate, which is
    # NOT a gapmer -- a gapmer is defined by modified wings. The ASO Atlas
    # training distribution is ~53% 5-10-5 MOE and ~30% 3-10-3 cEt, so the
    # default configuration sits outside it; pass wing_mod explicitly.
    bb = torch.full((1, L), K_VOC["PS"], dtype=torch.long)
    pad = torch.zeros(1, L, dtype=torch.bool)
    gr = torch.tensor([row], dtype=torch.long)

    used, cid = "均值嵌入回退（未收录细胞系）", None
    if cell_line:
        c2i = pack["cell2id"]
        hit_key = cell_line if cell_line in c2i else next(
            (k for k in c2i if k.lower() == str(cell_line).lower()), None)
        if hit_key is not None:
            cid, used = c2i[hit_key], hit_key
    cell_vec = (m.cell_emb(torch.tensor([cid], dtype=torch.long))
                if cid is not None else pack["cell_mean"].unsqueeze(0))
    dl = torch.tensor([pack["dose_mu"]], dtype=torch.float32)
    dm = torch.ones(1, dtype=torch.float32)

    with torch.no_grad():
        if cid is not None:
            pred = m(base, sug, bb, pad, gr,
                     torch.tensor([cid], dtype=torch.long), dl, dm).item()
        else:
            # Unknown cell line: reuse the module's own forward for everything
            # except the cell embedding, which is replaced by the mean of the
            # trained rows. Reimplementing the forward pass here (as pre-v4 did)
            # silently skipped any layer added to the model -- the positional
            # embedding among them.
            x = m.in_proj(torch.cat([m.base_emb(base), m.sugar_emb(sug),
                                     m.bb_emb(bb)], dim=-1))
            if getattr(m, "pos_emb", None) is not None:
                x = x + m.pos_emb(torch.arange(x.shape[1]).unsqueeze(0))
            x = m.encoder(x, src_key_padding_mask=pad)
            valid = (~pad).float().unsqueeze(-1)
            pooled = torch.cat([(x * valid).sum(1) / valid.sum(1).clamp(min=1.0),
                                x.masked_fill(pad.unsqueeze(-1), -1e4).max(1).values],
                               dim=-1)
            g = m.esm_proj(m.esm_table[gr])
            f = torch.cat([pooled, g, cell_vec, dl.unsqueeze(-1),
                           dm.unsqueeze(-1)], dim=-1)
            pred = m.head(f).squeeze(-1).item()

    # inhibition_pct and kd must agree. Pre-v4 clamped kd to a 0.20 floor while
    # leaving inhibition_pct free, so the same call could return
    # inhibition_pct=11.0 together with kd=0.200, and every prediction below 20%
    # collapsed onto one value, destroying the ranking at the low-potency end --
    # exactly the end a triage tool needs to resolve.
    inh = float(min(max(pred, 0.0), 95.0))
    kd = inh / 100.0
    arch = pack.get("arch", {})
    mm = pack.get("metrics", {})
    return dict(inhibition_pct=round(inh, 1), kd=round(kd, 3),
                cell_line_used=used, target=target_gene, raw_pred=round(pred, 2),
                chemistry=dict(wing=wing_eff, wing_len=wl,
                               backbone="PS (uniform)",
                               is_gapmer=bool(wl > 0 and wing_eff != "DNA"),
                               mapped=("LNA->cEt" if str(wing_mod or "").lower() == "lna"
                                       else None)),
                position_aware=bool(arch.get("use_pos_emb", False)),
                model=("efficacy ASO head (ASO Atlas gapmers); "
                       + ("position-aware" if arch.get("use_pos_emb")
                          else "PRE-v4 checkpoint: no positional embedding, so the "
                               "sequence branch is a bag of (base, sugar, backbone) "
                               "triples and cannot distinguish where a modification sits")),
                heldout_metrics=mm if mm else
                "checkpoint carries no held-out metrics; do not quote an accuracy for it")


if __name__ == "__main__":
    import json
    print(json.dumps(predict_inhibition(
        "TGCATCGTACGTAGCTGATC", "APOC3", cell_line="hepg2"),
        ensure_ascii=False, indent=2))


# ===== hybrid routing (OPTIONAL, NOT PART OF THIS REPOSITORY) =====
#
# `predict_kd_hybrid` can route known target genes to an external gradient-boosted
# model ("xgboost_v5") that lives in a separate, private platform repository
# reached through $RNA_ROBOT_HOME. That model is NOT distributed here: this
# repository contains no weights, no training script, no feature extractor and no
# evaluation for it, and it is absent from results/MANIFEST_sha256.txt and from
# the release assets.
#
# Consequences, stated plainly because they were previously only implied:
#   * With RNA_ROBOT_HOME unset -- the case for every external user -- the branch
#     below is unreachable and `predict_kd_hybrid` is a thin wrapper around
#     `predict_inhibition`. Prefer calling `predict_inhibition` directly.
#   * The comparison that motivated this routing (PCC 0.58 vs 0.51 on a
#     "patent split") is not reproducible from this repository, and that split
#     was grouped by ASO Atlas `custom_id`, i.e. by source patent TABLE rather
#     than by patent. Those numbers should not be cited as a head-to-head result.
_XGB_PROJ = os.environ.get("RNA_ROBOT_HOME", "")  # private platform root; unset by default
_XGB_STATE = {"ok": None}


def _xgb_available():
    if _XGB_STATE["ok"] is None:
        try:
            import xgboost  # noqa: F401
            _XGB_STATE["ok"] = os.path.isdir(_XGB_PROJ)
        except Exception:
            _XGB_STATE["ok"] = False
    return _XGB_STATE["ok"]


def _chem_stub(wing_eff, L, wl):
    """构造 train_aso_xgb.extract_chemistry_features 兼容的 chemistry stub（翼修饰语义）。"""
    class _Mod:
        def __init__(self, m, t, pos):
            self.__dict__ = {"modification": m, "type": t, "positions": pos}
    class _Chem:
        def __init__(self, d):
            self.__dict__ = {"__dict__": d}
    mods = []
    if wing_eff != "DNA":
        pos = list(range(1, wl + 1)) + list(range(L - wl + 1, L + 1))
        mods.append(_Mod(wing_eff, "sugar", pos))
    mods.append(_Mod("PS", "backbone", list(range(1, L + 1))))
    return _Chem({"length": L, "modifications": mods})


def predict_kd_hybrid(seq, target_gene, cell_line=None, wing_mod=None, wing_len=3):
    """统一效力入口：按靶基因是否在 XGBoost 词表内路由，返回字段与 predict_inhibition 一致
    （额外带 model_source: 'xgboost_v5' | 'transformer'）。"""
    base = predict_inhibition(seq, target_gene, cell_line=cell_line,
                              wing_mod=wing_mod, wing_len=wing_len)
    base["model_source"] = "transformer"
    if base.get("error") or not _xgb_available():
        return base
    try:
        if _XGB_PROJ not in sys.path:
            sys.path.insert(0, _XGB_PROJ)
        from utils import train_aso_xgb as _tx
        _tx._ensure_aso_model()
        gv = (_tx._aso_meta or {}).get("gene_vocab", {})
        if str(target_gene) not in gv:
            return base
        seq_clean = "".join(ch for ch in str(seq).upper() if ch in "ACGT")
        L = len(seq_clean)
        wing_eff = WING_MAP.get(str(wing_mod or "dna").lower(), "DNA")
        wl = max(0, min(int(wing_len), (L - 1) // 2))
        kd = _tx.predict_aso_kd(seq_clean, cell_line=cell_line or "HeLa",
                            target_gene=str(target_gene),
                            chemistry_obj=_chem_stub(wing_eff, L, wl))
        inh = float(min(max(kd * 100.0, 0.0), 95.0))
        kd_c = float(min(max(kd, 0.2), 0.95))
        return dict(inhibition_pct=round(inh, 1), kd=round(kd_c, 3),
                    cell_line_used=cell_line or "未指定",
                    target=target_gene, raw_pred=round(kd, 3),
                    chemistry=dict(wing=wing_eff, wing_len=wl,
                                   mapped=("LNA->cEt" if str(wing_mod or "").lower() == "lna"
                                           else None)),
                    model_source="xgboost_v5")
    except Exception as e:  # noqa
        print(f"[hybrid] xgb 分支失败回退 transformer: {str(e)[:90]}", flush=True)
        return base
