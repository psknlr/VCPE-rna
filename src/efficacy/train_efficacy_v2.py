"""Train the ASO efficacy head on ASO Atlas 2.0 with OFFICIAL patent-grouped 5-fold CV.

Features:
  - HELM-parsed per-position features (bases incl. 5meC/I/M/O/Q, sugars moe/cEt/d/r,
    backbones sp/po) on 144,767 fold-assigned rows
  - dose (log10 nM) + treatment period (log10 h) with missing flags
  - optional GENCODE region features (--use_region / --no_region)

Evaluation protocol (v4; supersedes v2/v2.5/v3):
  - outer loop = the official patent-grouped folds: train on 4, test on 1
  - epoch selection happens on a SCREEN-GROUPED inner validation split carved out
    of the 4 training folds; the held-out fold is touched exactly once, after
    training, at the selected epoch
  - per-screen Spearman is computed against the measured inhibition label
  - top-5% enrichment is normalised so that 1.0 == random
  - pooled and per-screen estimators are reported side by side, each with a
    percentile bootstrap CI; --seeds repeats the whole CV to expose run-to-run spread

Errata for previously published numbers (see `errata` in the output JSON):
  v2/v2.5/v3 selected the reported epoch on the test fold, computed per-screen
  Spearman against a feature tensor rather than the label, and scaled enrichment
  by 5 instead of 20. Those three defects are fixed here; numbers produced by
  this script are not comparable to the v2/v2.5/v3 result files.

CPU budget: 5 folds x 8 epochs x ~110s ≈ 1.5h per seed. Runs on GPU when available.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
torch.set_num_threads(8)
from scipy.stats import spearmanr, pearsonr

from model_efficacy import MAX_LEN

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "aso_atlas_2"
ESM_TABLE = HERE.parent.parent / "data" / "drive_weights" / \
    "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"

BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 4, "5meC": 5, "I": 6, "M": 7,
         "O": 8, "Q": 9, "N": 10}
SUGARS = {"d": 0, "moe": 1, "cet": 2, "r": 3, "other": 4}
BACKBONES = {"po": 0, "sp": 1, "other": 2}

ALIAS = {"ACC1": "ACACA", "ACC2": "ACACB", "HMGCR-B": "HMGCR"}
# NOTE: region-feature use is a CLI flag (--use_region / --no_region), not a
# module constant, so that the v3-vs-v2.5 ablation is reproducible without
# editing source. Releases up to v3 hard-coded this to True.
ALIAS_UP = {"C9ORF72": "C9orf72"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DATA / "in_vitro_clean.parquet"))
    p.add_argument("--esm_table", type=str, default=str(ESM_TABLE))
    p.add_argument("--out_dir", type=str, default=str(HERE.parent.parent / "results" / "efficacy_v2"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="run the whole CV once per seed and report across-seed spread; "
                        "overrides --seed when given")
    p.add_argument("--inner_val_frac", type=float, default=0.2,
                   help="fraction of TRAINING screens held out inside each fold for "
                        "epoch selection; the test fold is never used for selection")
    p.add_argument("--use_region", dest="use_region", action="store_true", default=True,
                   help="include GENCODE region features (v3)")
    p.add_argument("--no_region", dest="use_region", action="store_false",
                   help="ablate GENCODE region features (v2.5 configuration)")
    p.add_argument("--no_pos_emb", dest="use_pos_emb", action="store_false",
                   default=True,
                   help="ablate the positional embedding (pre-v4 bag-of-triples "
                        "sequence encoder); report this ablation rather than "
                        "assuming position matters")
    p.add_argument("--tag", type=str, default=None,
                   help="suffix for output files; defaults to v3/v2.5 by --use_region")
    return p.parse_args()


def encode(ls, base_seqs, sugar_seqs, bb_seqs):
    B = len(ls)
    base = np.zeros((B, MAX_LEN), dtype=np.int64)
    sug = np.zeros((B, MAX_LEN), dtype=np.int64)
    bb = np.zeros((B, MAX_LEN), dtype=np.int64)
    pad = np.ones((B, MAX_LEN), dtype=bool)
    for i, (L, s, su, b) in enumerate(zip(ls, base_seqs, sugar_seqs, bb_seqs)):
        bl, sul, bbl = s.split(","), su.split(","), b.split(",")
        for j in range(min(int(L), MAX_LEN)):
            base[i, j] = BASES.get(bl[j] if j < len(bl) else "N", BASES["N"])
            sug[i, j] = SUGARS.get(sul[j] if j < len(sul) else "other", SUGARS["other"])
            bb[i, j] = BACKBONES.get(bbl[j] if j < len(bbl) else "other", BACKBONES["other"])
            pad[i, j] = False
    return (torch.from_numpy(base), torch.from_numpy(sug), torch.from_numpy(bb),
            torch.from_numpy(pad))


class EfficacyV2(nn.Module):
    """Same architecture family as v1, widened vocabs + period/lineage features."""

    def __init__(self, esm_matrix, n_cell_lines, n_lineages, d=128, hidden=256, dropout=0.1,
                 use_region=False, use_pos_emb=True):
        super().__init__()
        esm_matrix = esm_matrix.float()
        # Non-persistent: the ESM2 table is a frozen copy of a public lookup
        # table, and persisting it added ~400 MB of redundant bytes to every
        # checkpoint written before v4.
        self.register_buffer("esm_table", esm_matrix, persistent=False)
        self.esm_dim = esm_matrix.shape[1]
        # No padding_idx: encode() leaves padded tail positions at index 0
        # (which is a REAL class here -- "A"/"d"/"po"), and those positions are
        # excluded by src_key_padding_mask. Pointing padding_idx at "N"/"other"
        # instead, as pre-v4 did, froze the catch-all classes at zero without
        # ever matching an actual pad position.
        self.base_emb = nn.Embedding(len(BASES), 16)
        self.sugar_emb = nn.Embedding(len(SUGARS), 16)
        self.bb_emb = nn.Embedding(len(BACKBONES), 16)
        self.in_proj = nn.Linear(48, d)
        # Positional embedding: without it the encoder is permutation-equivariant
        # and the pooling permutation-invariant, so the sequence branch reduces
        # to a bag of (base, sugar, backbone) triples and cannot represent where
        # a modification sits -- see model_efficacy.py for the full note.
        self.use_pos_emb = use_pos_emb
        self.pos_emb = nn.Embedding(MAX_LEN, d) if use_pos_emb else None
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.esm_proj = nn.Linear(self.esm_dim, d)
        self.cell_emb = nn.Embedding(n_cell_lines, 16)
        self.lin_emb = nn.Embedding(n_lineages, 8)
        self.use_region = use_region
        self.region_emb = nn.Embedding(5, 8) if use_region else None
        extra = 11 if use_region else 0  # region_emb(8) + match_any + rel_pos + occ_log
        # pooled(2d) + esm(d) + cell(16) + lineage(8) + dose(2) + period(2) [+ region 12]
        self.head = nn.Sequential(
            nn.Linear(2 * d + d + 16 + 8 + 4 + extra, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, base_ids, sugar_ids, bb_ids, pad_mask, gene_rows, cell_ids,
                lin_ids, dose_log, dose_missing, period_log, period_missing,
                region_ids=None, match_any=None, rel_pos=None, occ_log=None):
        x = self.in_proj(torch.cat([self.base_emb(base_ids), self.sugar_emb(sugar_ids),
                                    self.bb_emb(bb_ids)], dim=-1))
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(x.shape[1], device=x.device).unsqueeze(0))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = torch.cat([(x * valid).sum(1) / valid.sum(1).clamp(min=1.0),
                            x.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(1).values],
                           dim=-1)
        g = self.esm_proj(self.esm_table[gene_rows])
        feats = [pooled, g, self.cell_emb(cell_ids), self.lin_emb(lin_ids),
                 dose_log.unsqueeze(-1), dose_missing.unsqueeze(-1),
                 period_log.unsqueeze(-1), period_missing.unsqueeze(-1)]
        if self.use_region:
            feats += [self.region_emb(region_ids), match_any.unsqueeze(-1),
                      rel_pos.unsqueeze(-1), occ_log.unsqueeze(-1)]
        f = torch.cat(feats, dim=-1)
        return self.head(f).squeeze(-1)


# Metric definitions live in metrics.py (torch-free, unit-tested in tests/).
# They were previously inlined per script, which is how the per-screen and
# enrichment defects came to differ between v1 and v2.
from metrics import (  # noqa: E402
    bootstrap_ci, enrichment_top5, per_screen_spearman, regression_metrics)


def metrics(trues, preds, groups=None, with_ci=False, seed=0):
    return regression_metrics(trues, preds, groups=groups, with_ci=with_ci, seed=seed)


def run_fold(fold_k, tr, te, data_cache, args, device):
    import numpy as _np
    REGION = {"CDS": 0, "3UTR": 1, "5UTR": 2, "nc": 3, "no_match": 4}
    esm, gene2row, cell2id, lin2id, dose_mu, per_mu = data_cache
    torch.manual_seed(args.seed + fold_k)
    model = EfficacyV2(esm, n_cell_lines=len(cell2id), n_lineages=len(lin2id),
                       use_region=args.use_region,
                       use_pos_emb=args.use_pos_emb).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.HuberLoss()

    def prep(part):
        base, sug, bb, pad = encode(part["L"].tolist(), part["base_seq"].tolist(),
                                    part["sugar_seq"].tolist(), part["bb_seq"].tolist())
        gr = torch.tensor([gene2row.get(g, gene2row["UNKNOWN"])
                           for g in part["target_gene"]], dtype=torch.long)
        ci = torch.tensor([cell2id.get(c, cell2id["UNKNOWN"])
                           for c in part["cell_line"]], dtype=torch.long)
        li = torch.tensor([lin2id.get(l, lin2id["UNKNOWN"])
                           for l in part["lineage"]], dtype=torch.long)
        dl = torch.tensor(part["dosage_log10"].fillna(dose_mu).values, dtype=torch.float32)
        dm = torch.tensor(part["dosage_log10"].isna().values, dtype=torch.float32)
        pl = torch.tensor(part["period_log10"].fillna(per_mu).values, dtype=torch.float32)
        pm = torch.tensor(part["period_log10"].isna().values, dtype=torch.float32)
        out = [base, sug, bb, pad, gr, ci, li, dl, dm, pl, pm]
        if args.use_region:
            reg = torch.tensor([REGION.get(r, 4) for r in part["region"]], dtype=torch.long)
            ma = torch.tensor((part["n_enst_matched"] > 0).values, dtype=torch.float32)
            rp = torch.tensor(part["rel_pos"].fillna(0).values, dtype=torch.float32)
            oc = torch.tensor(np.log1p(part["n_occurrences"].fillna(0).values), dtype=torch.float32)
            out += [reg, ma, rp, oc]
        out.append(torch.tensor(part["inhibition"].values, dtype=torch.float32))
        return tuple(out)

    # ---- inner split for epoch selection (NEVER touch the test fold) ----
    # Grouped by patent screen so that no screen spans inner-train / inner-val;
    # this mirrors the outer official-fold grouping one level down.
    tr_groups_all = tr["group"].astype(str).values
    uniq_g = _np.unique(tr_groups_all)
    g_rng = _np.random.default_rng(args.seed + 1000 + fold_k)
    g_perm = g_rng.permutation(len(uniq_g))
    n_val_g = max(1, int(round(len(uniq_g) * args.inner_val_frac)))
    val_g = set(uniq_g[g_perm[:n_val_g]])
    is_val = _np.array([g in val_g for g in tr_groups_all])
    tr_in, tr_val = tr[~is_val], tr[is_val]
    print(f"  fold{fold_k} inner split: train {len(tr_in)} ({len(uniq_g)-n_val_g} screens) / "
          f"val {len(tr_val)} ({n_val_g} screens)", flush=True)

    def to_dev(t):
        return tuple(x.to(device) for x in t)

    tb_all = prep(tr_in)
    vb_all = prep(tr_val)
    sb_all = prep(te)
    y_dev, tb = tb_all[-1].to(device), to_dev(tb_all[:-1])
    y_val, vb = vb_all[-1].numpy(), to_dev(vb_all[:-1])
    y_te, sb = sb_all[-1].numpy(), to_dev(sb_all[:-1])
    te_groups = te["group"].astype(str).values
    n = len(y_dev)

    @torch.no_grad()
    def predict(batched):
        model.eval()
        outs = []
        total = batched[0].shape[0]
        for lo in range(0, total, args.eval_batch_size):
            hi = min(lo + args.eval_batch_size, total)
            outs.append(model(*[t[lo:hi] for t in batched]).cpu().numpy())
        return _np.concatenate(outs)

    best_val, best_ep, best_state = -_np.inf, -1, None
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=device)
        run = []
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            out = model(*[t[idx] for t in tb])
            loss = loss_fn(out, y_dev[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(loss.item()))
        sched.step()
        v_rho = float(spearmanr(y_val, predict(vb)).statistic)
        history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(run)),
                            inner_val_spearman=v_rho))
        if v_rho > best_val:
            best_val, best_ep = v_rho, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  fold{fold_k} ep{epoch+1}: loss {np.mean(run):.3f} | "
              f"inner-val spearman {v_rho:.4f}{' *' if best_ep == epoch + 1 else ''}"
              f" | {time.time()-t0:.0f}s", flush=True)

    # ---- single, final touch of the held-out fold, at the selected epoch ----
    if best_state is not None:
        model.load_state_dict(best_state)
    preds = predict(sb)
    m = metrics(y_te, preds, groups=te_groups, with_ci=True, seed=args.seed + fold_k)
    m["selected_epoch"] = best_ep
    m["inner_val_spearman_at_selection"] = best_val
    m["history"] = history

    _np.savez_compressed(Path(args.out_dir) / f"fold{fold_k}_preds.npz",
                         preds=preds, trues=y_te,
                         groups=te_groups, helm=te["helm"].values)
    print(f"  fold{fold_k} TEST (ep{best_ep}): spearman {m['spearman']:.4f} "
          f"CI95 [{m['spearman_ci95'][0]:.4f}, {m['spearman_ci95'][1]:.4f}] | "
          f"per-screen median {m['per_screen_median']} (n={m['n_screens']}) | "
          f"enrich {m['enrich_top5']:.2f}x", flush=True)
    return m


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)

    df = pd.read_parquet(args.data)
    df = df[df["fold"].notna()].copy()
    print(f"fold-assigned rows: {len(df)}", flush=True)

    # v2.5 target-gene cleanup
    GARBAGE = re.compile(r"^(exon|intron|utr|aug|3'|5'|5′|i?res|orf\s*$)", re.I)
    def clean_gene(g):
        if not isinstance(g, str):
            return None
        g = g.strip()
        if GARBAGE.match(g) or "UTR" in g.upper() or g.upper().startswith("EXON"):
            return None
        if g in ALIAS:
            return ALIAS[g]
        if g.upper() in ALIAS_UP:
            return ALIAS_UP[g.upper()]
        return g
    df["target_RNA"] = df["target_RNA"].map(clean_gene)
    n_before = len(df)
    df = df[df["target_RNA"].notna()]
    print(f"[cleanup] dropped garbage-target rows: {n_before - len(df)} | "
          f"rows kept {len(df)}", flush=True)

    # helm_seq: join bases back (per-position base string)
    df["helm_seq"] = df["base_seq"].str.split(",").str.join("")

    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab.keys()) if isinstance(tab, dict) else []
    esm = torch.stack([v.float() for v in tab.values()]) if isinstance(tab, dict) else tab
    gene2row = {s: i for i, s in enumerate(symbols)}
    gene2row.setdefault("UNKNOWN", 0)

    # v3 region features, aligned on the post-cleanup df.index.
    # region_features.parquet is generated against the UNFILTERED in_vitro_clean
    # table, so this is a positional join that silently misaligns if either file
    # is regenerated alone. Guard it rather than trusting index arithmetic.
    feat = pd.read_parquet(DATA.parent / "aso_atlas_2" / "region_features.parquet")
    missing_idx = df.index.difference(feat.index)
    if len(missing_idx):
        raise RuntimeError(
            f"region_features.parquet is missing {len(missing_idx)} of the {len(df)} "
            "rows required after target-gene cleanup; regenerate it from the current "
            "in_vitro_clean.parquet (src/efficacy/region_features.py) before training.")
    feat = feat.loc[df.index]
    if "helm" in feat.columns and "helm" in df.columns:
        mism = int((feat["helm"].values != df["helm"].values).sum())
        if mism:
            raise RuntimeError(
                f"region_features.parquet is misaligned with in_vitro_clean.parquet: "
                f"{mism} HELM keys differ after .loc alignment.")
        feat = feat.drop(columns=["helm"])
    df = pd.concat([df.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    df["target_gene"] = df["target_RNA"].fillna("UNKNOWN").astype(str)
    # fillna on the frame itself, so that the vocab built below and the lookups
    # performed in prep() see the same values (previously the vocab contained
    # "UNKNOWN" only when NaNs were present, making cell2id["UNKNOWN"] a KeyError
    # on NaN-free data).
    df["cell_line"] = df["cell_line"].fillna("UNKNOWN").astype(str)
    df["lineage"] = (df["lineage"] if "lineage" in df.columns else "UNKNOWN")
    df["lineage"] = pd.Series(df["lineage"], index=df.index).fillna("UNKNOWN").astype(str)
    cell2id = {c: i for i, c in enumerate(sorted(df["cell_line"].unique()))}
    cell2id.setdefault("UNKNOWN", len(cell2id))
    lin2id = {l: i for i, l in enumerate(sorted(df["lineage"].unique()))}
    lin2id.setdefault("UNKNOWN", len(lin2id))
    dose_mu = float(df["dosage_log10"].dropna().mean())
    per_mu = float(df["period_log10"].dropna().mean())
    data_cache = (esm, gene2row, cell2id, lin2id, dose_mu, per_mu)
    print(f"params context: cells={len(cell2id)} lineages={len(lin2id)} "
          f"genes-with-esm={sum(1 for g in df['target_gene'].unique() if g in gene2row)}"
          f"/{df['target_gene'].nunique()}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("v4_region" if args.use_region else "v4_noregion")
    seeds = args.seeds if args.seeds else [args.seed]
    t_all = time.time()

    per_seed = {}
    for s in seeds:
        args.seed = s
        torch.manual_seed(s)
        np.random.seed(s)
        print(f"\n===== seed {s} =====", flush=True)
        results, pooled_t, pooled_p, pooled_g = {}, [], [], []
        for k in sorted(df["fold"].unique()):
            tr = df[df["fold"] != k]
            te = df[df["fold"] == k]
            print(f"== fold {k}: train {len(tr)} / test {len(te)} ==", flush=True)
            m = run_fold(int(k), tr, te, data_cache, args, device)
            results[int(k)] = m
            z = np.load(out_dir / f"fold{int(k)}_preds.npz", allow_pickle=True)
            pooled_t.append(z["trues"]); pooled_p.append(z["preds"]); pooled_g.append(z["groups"])
        pt, pp, pg = (np.concatenate(pooled_t), np.concatenate(pooled_p),
                      np.concatenate(pooled_g))
        # Two distinct estimators, reported side by side and never subtracted
        # from one another: `pooled_*` ranks all held-out rows together, while
        # `per_screen_*` is the within-screen estimator OligoAI reports.
        pooled = metrics(pt, pp, groups=pg, with_ci=True, seed=s)
        fold_rhos = [results[k]["spearman"] for k in results]
        fold_ps = [results[k]["per_screen_median"] for k in results
                   if results[k]["per_screen_median"] is not None]
        per_seed[s] = dict(
            per_fold=results,
            pooled_over_all_folds=pooled,
            mean_of_fold_spearman=float(np.mean(fold_rhos)),
            sd_of_fold_spearman=float(np.std(fold_rhos, ddof=1)) if len(fold_rhos) > 1 else None,
            mean_of_fold_per_screen_median=float(np.mean(fold_ps)) if fold_ps else None,
            sd_of_fold_per_screen_median=(float(np.std(fold_ps, ddof=1))
                                          if len(fold_ps) > 1 else None),
        )
        print(f"-- seed {s}: pooled spearman {pooled['spearman']:.4f} "
              f"CI95 [{pooled['spearman_ci95'][0]:.4f}, {pooled['spearman_ci95'][1]:.4f}] | "
              f"per-screen median {pooled['per_screen_median']:.4f} "
              f"(n={pooled['n_screens']} screens)", flush=True)

    seed_pooled = [per_seed[s]["pooled_over_all_folds"]["spearman"] for s in seeds]
    seed_screen = [per_seed[s]["pooled_over_all_folds"]["per_screen_median"] for s in seeds]
    summary = dict(
        version=f"v4 ({tag}): inner-val epoch selection, label-correct per-screen, "
                f"enrichment normalised to 1.0=random, bootstrap CI, {len(seeds)} seed(s)",
        config=dict(use_region=args.use_region, use_pos_emb=args.use_pos_emb,
                    epochs=args.epochs, lr=args.lr,
                    inner_val_frac=args.inner_val_frac, seeds=seeds),
        per_seed=per_seed,
        across_seeds=dict(
            pooled_spearman_mean=float(np.mean(seed_pooled)),
            pooled_spearman_sd=float(np.std(seed_pooled, ddof=1)) if len(seeds) > 1 else None,
            per_screen_median_mean=float(np.mean(seed_screen)),
            per_screen_median_sd=float(np.std(seed_screen, ddof=1)) if len(seeds) > 1 else None,
        ),
        comparator=dict(
            note="literature values, NOT reproduced in this repository; different "
                 "preprocessing and screen counts, so these are context only and "
                 "no head-to-head claim is supported by them",
            OligoAI_per_screen_median="0.419 [0.290-0.533] over 299 screens",
            ASOptimizer_per_screen_median="0.076",
            OligoWalk_per_screen_median="0.147",
        ),
        errata=[
            "v2/v2.5/v3 selected the reported epoch on the held-out fold itself "
            "(best-over-epochs); those numbers are optimistically biased.",
            "v2/v2.5/v3 computed per-screen Spearman against the last feature "
            "tensor instead of the label; those per-screen numbers are void.",
            "v2/v2.5/v3 scaled top-5% enrichment by 5 instead of 20, so the "
            "random baseline for those values is 0.25, not 1.0.",
        ],
    )
    print(f"\n✅ CV DONE in {(time.time()-t_all)/60:.0f} min | "
          f"pooled Spearman {summary['across_seeds']['pooled_spearman_mean']:.4f} | "
          f"per-screen median {summary['across_seeds']['per_screen_median_mean']:.4f}",
          flush=True)
    with open(out_dir / f"cv_results_{tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   wrote {out_dir / f'cv_results_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main()
