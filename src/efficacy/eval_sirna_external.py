"""siRNA efficacy head v1 — EXTERNAL held-out eval (Takayuki Ichihara 2007 set).

训练：Huesken (ichihara_2007_1, 2,431 guide strands, 30 targets) 全量，架构/超参
与 train_sirna_v1.py 完全一致（21mer guide transformer + 冻结 ESM2 靶轴, 60ep/bs256/lr1e-3）。

外部验证：ichihara_2007_2（OligoGym 过滤版 Takayuki Ichihara 集, 419 条,
12 靶点 EGFR/TP53/GAPDH/... 与训练集零靶点重叠——真正的 external held-out。
RNAGenesis 论文用原始 702 条版本，此处为 OligoGym CC-BY 同源子集）。

指标：pooled Spearman/Pearson + per-target Spearman（排名指标，跨实验批次稳健）。

Run locally (CPU, minutes): python eval_sirna_external.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "..", "data", "oligogym", "ichihara_2007_1.csv.gz")
TEST_CSV = os.path.join(HERE, "..", "..", "data", "oligogym", "ichihara_2007_2.csv.gz")
ESM = os.path.join(HERE, "..", "..", "data", "drive_weights",
                   "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
OUT = os.path.join(HERE, "..", "..", "results", "sirna_v1")
BASES = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
MAX_LEN = 24
SEED = 0
EPOCHS, BS, LR = 60, 256, 1e-3


def parse_guide(fasta_field):
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


class SiRNANet(torch.nn.Module):
    def __init__(self, esm_matrix, d=96, hidden=192, dropout=0.1):
        super().__init__()
        self.register_buffer("esm_table", esm_matrix)   # [V, 5120] frozen
        self.esm_dim = esm_matrix.shape[-1]
        self.base_emb = torch.nn.Embedding(5, 24, padding_idx=4)
        self.pos_emb = torch.nn.Embedding(MAX_LEN, 24)
        layer = torch.nn.TransformerEncoderLayer(24, 4, dim_feedforward=96,
                                                dropout=dropout, batch_first=True,
                                                norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, 2)
        self.proj_t = torch.nn.Linear(self.esm_dim, 64)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(48 + 64, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))

    def forward(self, ids, pad, tgt_rows):
        x = self.base_emb(ids) + self.pos_emb.weight[: ids.shape[1]].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=pad)
        pooled = torch.cat([x.mean(1), x.max(dim=1).values], dim=-1)
        g = self.proj_t(self.esm_table[tgt_rows])
        return self.head(torch.cat([pooled, g], dim=-1)).squeeze(-1)


def metrics(y, p):
    rho = spearmanr(y, p).statistic if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    pr = np.corrcoef(y, p)[0, 1] if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    return dict(spearman=float(rho), pearson=float(pr))


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T0 = time.time()

    # ---- 数据 ----
    tr_df = pd.read_csv(TRAIN_CSV)
    te_df = pd.read_csv(TEST_CSV)
    n0 = len(te_df)
    def _has_target(col):
        s_ = col.astype(str).str.strip()
        return col.notna() & (s_.str.len() > 0) & \
            (~s_.str.lower().isin(["nan", "none", "na", "null", "-"]))
    te_df = te_df[_has_target(te_df["targets"])].copy()
    tr_df = tr_df[_has_target(tr_df["targets"])].copy()
    print(f"train: {len(tr_df)} rows / {tr_df['targets'].nunique()} targets | "
          f"test: {len(te_df)}/{n0} rows (dropped {n0-len(te_df)} empty-target) / "
          f"{te_df['targets'].nunique()} targets", flush=True)

    tab = torch.load(ESM, map_location="cpu", weights_only=False)
    sym2row = {s: i for i, s in enumerate(tab.keys())}
    esm_matrix = torch.stack(list(tab.values())).float()

    def lookup(t):
        if t in sym2row:
            return sym2row[t]
        if t + "1" in sym2row:          # HGNC 2020 renames, e.g. AARS -> AARS1
            return sym2row[t + "1"]
        return -1

    def build(df, name):
        guides = [parse_guide(f) for f in df["fasta"]]
        ids, pad = encode(guides)
        y = df["y"].values.astype(np.float32)
        targets = df["targets"].astype(str).values
        seqs = np.array([str(g) for g in guides])
        rows = np.array([lookup(t) for t in targets], dtype=np.int64)
        keep = rows >= 0
        n_drop = int((~keep).sum())
        if n_drop:
            # Previously these fell back to ESM row 0, i.e. some arbitrary real
            # gene's embedding stood in for an unmappable target.
            print(f"[esm/{name}] dropping {n_drop} rows with unmappable targets: "
                  f"{sorted(set(targets[~keep]))}", flush=True)
        return dict(ids=torch.from_numpy(ids[keep]), pad=torch.from_numpy(pad[keep]),
                    y=y[keep], targets=targets[keep], seqs=seqs[keep],
                    rows=torch.from_numpy(rows[keep]), n_dropped_unmappable=n_drop)

    TR, TE = build(tr_df, "train"), build(te_df, "test")

    # Explicit held-out check. The docstring asserted zero target overlap;
    # nothing verified it, and the two files come from one literature family.
    tgt_ov = sorted(set(TE["targets"]) & set(TR["targets"]))
    seq_ov = sorted(set(TE["seqs"]) & set(TR["seqs"]))
    n_te_seqs = len(set(TE["seqs"]))
    print(f"[holdout-check] target overlap: {len(tgt_ov)} {tgt_ov[:10]} | "
          f"exact guide-sequence overlap: {len(seq_ov)}/{n_te_seqs}", flush=True)
    if tgt_ov or seq_ov:
        print("[holdout-check] WARNING: this is no longer a clean external "
              "evaluation -- overlapping targets/sequences must be removed or "
              "reported alongside the headline number.", flush=True)
    covered = [t for t in set(TE["targets"]) if t in sym2row or t + "1" in sym2row]
    print(f"test targets mapped to ESM2: {len(covered)}/{len(set(TE['targets']))} "
          f"-> {sorted(covered)}", flush=True)

    # ---- 训练（全量 Huesken，同 v1 超参）----
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    model = SiRNANet(esm_matrix).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    y_mu, y_sd = TR["y"].mean(), TR["y"].std() + 1e-6
    n = len(TR["y"])
    tr_ids, tr_pad, tr_rows = TR["ids"].to(dev), TR["pad"].to(dev), TR["rows"].to(dev)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(n)
        for lo in range(0, n, BS):
            idx = torch.from_numpy(perm[lo:lo + BS])
            out = model(tr_ids[idx], tr_pad[idx], tr_rows[idx])
            yb = torch.from_numpy((TR["y"][perm[lo:lo + BS]] - y_mu) / y_sd).float().to(dev)
            loss = torch.nn.functional.huber_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                p_tr = model(tr_ids, tr_pad, tr_rows).cpu().numpy() * y_sd + y_mu
            print(f"ep{ep+1}: train spearman {metrics(TR['y'], p_tr)['spearman']:.4f} | "
                  f"{time.time()-T0:.0f}s", flush=True)

    torch.save(model.state_dict(), os.path.join(OUT, "sirna_v1_full_train.pt"))

    # ---- 外部评估 ----
    model.eval()
    with torch.no_grad():
        p_te = model(TE["ids"].to(dev), TE["pad"].to(dev),
                     TE["rows"].to(dev)).cpu().numpy() * y_sd + y_mu
    pooled = metrics(TE["y"], p_te)
    rng_ci = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        i = rng_ci.integers(0, len(TE["y"]), len(TE["y"]))
        if np.std(TE["y"][i]) > 1e-12 and np.std(p_te[i]) > 1e-12:
            v = spearmanr(TE["y"][i], p_te[i]).statistic
            if np.isfinite(v):
                boot.append(v)
    pooled["spearman_ci95"] = [float(np.quantile(boot, 0.025)),
                               float(np.quantile(boot, 0.975))] if boot else [float("nan")] * 2
    per_t, per_t_n = {}, {}
    for t in sorted(set(TE["targets"])):
        m = np.array([i for i, x in enumerate(TE["targets"]) if x == t])
        per_t[t] = metrics(TE["y"][m], p_te[m])["spearman"]
        per_t_n[t] = int(len(m))
    med_t = float(np.median([v for v in per_t.values() if np.isfinite(v)]))

    summary = {
        "model": "sirna v1 (Huesken full-train) -> external Takayuki/Ichihara_2007_2",
        "protocol": ("fixed epoch budget, no test-based early stopping; the test set "
                     "is scored exactly once. Hyperparameters are inherited from "
                     "train_sirna_v1.py, whose pre-v4 tuning used test-fold epoch "
                     "selection -- so this evaluation is held-out at the level of "
                     "weights but not fully independent of the test data at the "
                     "level of hyperparameter choice."),
        "train": {"rows": int(n), "targets": int(tr_df["targets"].nunique()),
                  "dropped_unmappable": int(TR["n_dropped_unmappable"])},
        "test": {"rows": int(len(TE["y"])), "rows_in_file": int(n0),
                 "dropped_missing_target": int(n0 - len(te_df)),
                 "dropped_unmappable": int(TE["n_dropped_unmappable"]),
                 "targets": int(len(set(TE["targets"]))), "esm2_covered": len(covered)},
        "holdout_check": {"target_overlap_with_train": tgt_ov,
                          "n_exact_guide_sequence_overlap": len(seq_ov),
                          "n_unique_test_guides": n_te_seqs},
        "pooled": pooled,
        "per_target_spearman": per_t,
        "per_target_n": per_t_n,
        "per_target_median": med_t,
        "per_target_median_caveat": ("~35 rows per target gives a per-target "
                                     "Spearman standard error of roughly +/-0.15; "
                                     "read the median with that in mind."),
    }
    json.dump(summary, open(os.path.join(OUT, "external_ichihara2.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n✅ EXTERNAL DONE {time.time()-T0:.0f}s | pooled spearman "
          f"{pooled['spearman']:.4f} / pearson {pooled['pearson']:.4f} | "
          f"per-target median {med_t:.4f}", flush=True)
    for t, v in per_t.items():
        print(f"   {t:12s} {v:+.3f}")


if __name__ == "__main__":
    main()
