"""siRNA efficacy head v1 — Huesken (ichihara_2007_1, 2,431 guide strands, 30 targets).

Architecture: per-position base embedding (21mer antisense/guide strand) -> 2-layer
transformer -> mean+max pool -> ESM2 target-gene embedding (frozen) -> MLP regression.
No sugar/backbone/region features (siRNA is unmodified 21mer in this dataset);
no cell_line/dose (single-protocol experiment).

Eval: target-grouped 5-fold CV (no leakage across targets), reporting pooled
Spearman + per-target median Spearman (same protocol style as the ASO head).

Run locally (CPU, minutes): python train_sirna_v1.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "oligogym", "ichihara_2007_1.csv.gz")
ESM = os.path.join(HERE, "..", "..", "data", "drive_weights",
                   "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
OUT = os.path.join(HERE, "..", "..", "results", "sirna_v1")
BASES = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
MAX_LEN = 24
SEED = 0


def parse_guide(fasta_field):
    """fasta col = 'sense.antissent(guide)' — RNA2 (2nd segment) is the antisense guide."""
    parts = str(fasta_field).split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def encode(seqs):
    B = len(seqs)
    ids = np.zeros((B, MAX_LEN), dtype=np.int64)
    pad = np.ones((B, MAX_LEN), dtype=bool)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(str(s)[:MAX_LEN]):
            ids[i, j] = BASES.get(ch.upper(), 4)
            pad[i, j] = False
    return ids, pad


class SiRNANet(nn.Module):
    def __init__(self, esm_matrix, n_targets_map, d=96, hidden=192, dropout=0.1):
        super().__init__()
        self.register_buffer("esm_table", esm_matrix)   # [V, 5120] frozen
        self.esm_dim = esm_matrix.shape[-1]
        self.base_emb = nn.Embedding(5, 24, padding_idx=4)
        self.pos_emb = nn.Embedding(MAX_LEN, 24)
        layer = nn.TransformerEncoderLayer(24, 4, dim_feedforward=96,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.tgt_lookup = n_targets_map                # dict target -> esm row
        self.proj_t = nn.Linear(self.esm_dim, 64)
        self.head = nn.Sequential(
            nn.Linear(48 + 64, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, ids, pad, tgt_rows):
        x = self.base_emb(ids) + self.pos_emb.weight[: ids.shape[1]].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=pad)
        pooled = torch.cat([x.mean(1), x.max(dim=1).values], dim=-1)   # [B, 48]
        g = self.proj_t(self.esm_table[tgt_rows])                       # [B, 64]
        return self.head(torch.cat([pooled, g], dim=-1)).squeeze(-1)


def metrics(y, p):
    rho = spearmanr(y, p).statistic if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    pr = np.corrcoef(y, p)[0, 1] if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    return dict(spearman=float(rho), pearson=float(pr))


def bootstrap_ci(y, p, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    if n < 8:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(y[i]) < 1e-12 or np.std(p[i]) < 1e-12:
            continue
        v = spearmanr(y[i], p[i]).statistic
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def run_fold(k, tr, te, X, Y, T, targets, dev, epochs=60, bs=256, lr=1e-3,
             inner_val_frac=0.2):
    """Train on `tr`, select the epoch on a target-grouped inner split of `tr`,
    then score `te` exactly once at the selected epoch.

    The previous implementation tracked `best = max(best, spearman(te))` across
    epochs and returned that epoch's predictions, i.e. it selected the model on
    the very fold it reported. Every v1 siRNA CV number published from it
    (pooled 0.607, per-fold-best mean 0.6387) carries that bias.
    """
    torch.manual_seed(SEED + k)
    # inner split by target, so no target spans inner-train / inner-val
    tr_t = np.array([targets[i] for i in tr])
    uniq = np.unique(tr_t)
    g_rng = np.random.default_rng(SEED + 500 + k)
    n_val = max(1, int(round(len(uniq) * inner_val_frac)))
    val_t = set(uniq[g_rng.permutation(len(uniq))[:n_val]])
    is_val = np.array([t in val_t for t in tr_t])
    tr_in, tr_val = tr[~is_val], tr[is_val]

    model = SiRNANet(X["esm"], X["tgt_map"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    y_mu, y_sd = Y[tr_in].mean(), Y[tr_in].std() + 1e-6
    n = len(tr_in)
    rng = np.random.default_rng(SEED + k)

    @torch.no_grad()
    def predict(sel):
        model.eval()
        return model(X["ids"][sel].to(dev), X["pad"][sel].to(dev),
                     T[sel].to(dev)).cpu().numpy() * y_sd + y_mu

    best_val, best_ep, best_state = -np.inf, -1, None
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        run = []
        for lo in range(0, n, bs):
            idx = torch.from_numpy(perm[lo:lo + bs])
            sel = tr_in[idx]
            out = model(X["ids"][sel].to(dev), X["pad"][sel].to(dev), T[sel].to(dev))
            loss = nn.functional.huber_loss(
                out, torch.from_numpy(((Y[sel] - y_mu) / y_sd)).float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            run.append(loss.item())
        v = metrics(Y[tr_val], predict(tr_val))["spearman"]
        if v > best_val:
            best_val, best_ep = v, ep + 1
            best_state = {kk: vv.detach().cpu().clone()
                          for kk, vv in model.state_dict().items()}
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  fold{k} ep{ep+1}: loss {np.mean(run):.3f} | "
                  f"inner-val spearman {v:.4f} | {time.time()-T0:.0f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    p = predict(te)
    m = metrics(Y[te], p)
    m["spearman_ci95"] = list(bootstrap_ci(Y[te], p, seed=SEED + k))
    m["selected_epoch"] = best_ep
    m["inner_val_spearman_at_selection"] = float(best_val)
    return m, p


T0 = time.time()


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA)
    print(f"rows: {len(df)} | targets: {df['targets'].nunique()}", flush=True)

    guides = [parse_guide(f) for f in df["fasta"]]
    ids, pad = encode(guides)
    y = df["y"].values.astype(np.float32)
    targets = df["targets"].astype(str).values

    tab = torch.load(ESM, map_location="cpu", weights_only=False)
    sym2row = {s: i for i, s in enumerate(tab.keys())}
    esm_matrix = torch.stack(list(tab.values())).float()

    # Unmapped targets previously fell back to ESM row 0, silently handing the
    # model some arbitrary real gene's embedding. Drop those rows instead, and
    # say how many were dropped.
    def lookup(t):
        if t in sym2row:
            return sym2row[t]
        if t + "1" in sym2row:  # HGNC 2020 renames, e.g. AARS -> AARS1
            return sym2row[t + "1"]
        return -1
    tgt_rows = np.array([lookup(t) for t in targets], dtype=np.int64)
    keep = tgt_rows >= 0
    n_drop = int((~keep).sum())
    if n_drop:
        dropped = sorted(set(targets[~keep]))
        print(f"[esm] dropping {n_drop} rows with unmappable targets "
              f"({len(dropped)} symbols: {dropped[:8]}{'...' if len(dropped) > 8 else ''})",
              flush=True)
    ids, pad, y, targets, tgt_rows = (ids[keep], pad[keep], y[keep],
                                      targets[keep], tgt_rows[keep])
    print(f"targets mapped to ESM2: {len(set(targets))}/{df['targets'].nunique()} "
          f"| rows kept {len(y)}/{len(df)}", flush=True)

    X = dict(ids=torch.from_numpy(ids), pad=torch.from_numpy(pad),
             esm=esm_matrix, tgt_map={})
    T = torch.from_numpy(tgt_rows)

    # target-grouped 5-fold CV
    rng = np.random.default_rng(SEED)
    uniq_t = sorted(set(targets))
    rng.shuffle(uniq_t)
    folds = np.array_split(np.arange(len(uniq_t)), 5)
    per_fold = {}
    all_true, all_pred = [], []
    for k, ft in enumerate(folds, 1):
        te_t = set(uniq_t[i] for i in ft)
        te = np.array([i for i, t in enumerate(targets) if t in te_t])
        tr = np.array([i for i, t in enumerate(targets) if t not in te_t])
        m, p = run_fold(k, tr, te, X, y, T, targets, dev)
        per_fold[k] = dict(m, te_targets=len(te_t), n_te=int(len(te)))
        all_true.append(y[te]); all_pred.append(p)
        print(f"fold{k}: TEST spearman = {m['spearman']:.4f} "
              f"CI95 [{m['spearman_ci95'][0]:.4f}, {m['spearman_ci95'][1]:.4f}] "
              f"(ep{m['selected_epoch']}) | te_targets={len(te_t)} n_te={len(te)}",
              flush=True)
    tt, pp = np.concatenate(all_true), np.concatenate(all_pred)
    pooled = metrics(tt, pp)
    pooled["spearman_ci95"] = list(bootstrap_ci(tt, pp, seed=SEED))
    fold_rhos = [per_fold[k]["spearman"] for k in per_fold]
    summary = dict(
        model="sirna v1 (guide 21mer + ESM2 target axis, target-grouped 5-fold)",
        protocol=("epoch selected on a target-grouped inner split of the training "
                  "folds; each test fold scored once at the selected epoch"),
        pooled=pooled,
        per_fold=per_fold,
        mean_of_fold_spearman=float(np.mean(fold_rhos)),
        sd_of_fold_spearman=float(np.std(fold_rhos, ddof=1)),
        comparator=dict(
            note=("RNAGenesis reports a Huesken benchmark in its Fig 3h; no numeric "
                  "value is reproduced in this repository and no shared split or "
                  "preprocessing exists, so no 'matches/beats' claim is supported."),
            RNAGenesis_Huesken="not reproduced here",
        ),
        errata=["v1 selected the reported epoch on each held-out fold and pooled "
                "that epoch's predictions; the previously published pooled 0.607 "
                "and per-fold-best mean 0.6387 are optimistically biased."],
    )
    json.dump(summary, open(os.path.join(OUT, "cv_results.json"), "w"), indent=2)
    print(f"\n✅ CV DONE in {time.time()-T0:.0f}s | pooled spearman "
          f"{pooled['spearman']:.4f} CI95 [{pooled['spearman_ci95'][0]:.4f}, "
          f"{pooled['spearman_ci95'][1]:.4f}] | mean-of-folds "
          f"{np.mean(fold_rhos):.4f} ± {np.std(fold_rhos, ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
