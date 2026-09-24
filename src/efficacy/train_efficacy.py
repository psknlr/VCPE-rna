"""Train the ASO efficacy head v1 on ASO Atlas (190,927 rows).

Four disjoint row sets, built by grouped split (see build_split):
  train       : model fitting
  inner_val   : epoch/checkpoint selection ONLY -- never reported as performance
  group_val   : rows from held-out groups of --group_col, reported once at the
                selected epoch. NOTE: the default group_col `custom_id` is the
                ASO Atlas source patent TABLE, not the patent, so this is not a
                patent-level split unless a patent column is supplied.
  gene_holdout: all rows of ~15 unseen target genes, reported once at the
                selected epoch (tests ESM2-mechanism extrapolation)

Metrics: Spearman (primary) with percentile bootstrap CI, Pearson, and top-5%
enrichment normalised so that 1.0 == random.

Errata for previously published v1 numbers: the checkpoint was selected on
group_val and then group_val was reported as the headline ("val spearman 0.50"),
which is selection bias; and enrichment was scaled by 5 rather than 20, so the
random baseline for those logged values is 0.25, not 1.0. On that corrected
scale the published gene-holdout enrich_top5 of 0.2441 is ~0.98x random, i.e.
top-5% selection on unseen target genes was indistinguishable from chance.

Runs on CPU in ~15-30 min (small model); uses CUDA when available.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

from model_efficacy import EfficacyHead, BASES, SUGARS, BACKBONES, MAX_LEN

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "aso_atlas"
ESM_TABLE = HERE.parent.parent / "data" / "drive_weights" / \
    "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DATA / "aso_atlas_clean.parquet"))
    p.add_argument("--esm_table", type=str, default=str(ESM_TABLE))
    p.add_argument("--out_dir", type=str, default=str(HERE.parent.parent / "results" / "efficacy_v1"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_holdout_genes", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_pos_emb", dest="use_pos_emb", action="store_false",
                   default=True,
                   help="ablate the positional embedding, reproducing the pre-v4 "
                        "bag-of-triples sequence encoder")
    p.add_argument("--group_col", type=str, default="custom_id",
                   help="column defining the held-out group. ASO Atlas's "
                        "custom_id is the source patent TABLE, not the patent; "
                        "pass a true patent column here for a patent-level split.")
    return p.parse_args()


def encode_batch(seqs, sugar_seqs, bb_seqs):
    B = len(seqs)
    base = np.zeros((B, MAX_LEN), dtype=np.int64)
    sug = np.zeros((B, MAX_LEN), dtype=np.int64)
    bb = np.zeros((B, MAX_LEN), dtype=np.int64)
    pad = np.ones((B, MAX_LEN), dtype=bool)
    # Arrays start at 0, which is the reserved PAD index in the v4 vocabularies,
    # so unwritten tail positions are padding by construction. Unknown symbols
    # map to their explicit catch-all class, never to PAD and never to a
    # positionally-adjacent real class (pre-v4 this used bare literals 4/0/0,
    # which under the v4 vocabularies would mean "T"/PAD/PAD).
    for i, (s, su, bb_s) in enumerate(zip(seqs, sugar_seqs, bb_seqs)):
        L = min(len(s), MAX_LEN)
        su_l, bb_l = su.split(","), bb_s.split(",")
        for j in range(L):
            base[i, j] = BASES.get(s[j], BASES["N"])
            sug[i, j] = SUGARS.get(su_l[j] if j < len(su_l) else "other", SUGARS["other"])
            bb[i, j] = BACKBONES.get(bb_l[j] if j < len(bb_l) else "PO", BACKBONES["PO"])
            pad[i, j] = False
    return (torch.from_numpy(base), torch.from_numpy(sug), torch.from_numpy(bb),
            torch.from_numpy(pad))


def build_split(df, seed, n_holdout_genes, group_col="custom_id", inner_val_frac=0.15):
    """Grouped split: inner-val (model selection) + group-val + gene-holdout (both reported).

    `group_col` defaults to `custom_id`, which ASO Atlas defines as the SOURCE
    PATENT TABLE, not the patent. A single patent commonly contributes several
    tables, so a custom_id split does NOT guarantee patent-level separation and
    must not be described as a "patent split" -- pass the true patent column via
    --group_col when the input table carries one.

    Three disjoint row sets are returned plus an inner validation set carved out
    of training groups. Epoch/checkpoint selection uses ONLY `inner_val`; both
    `val_group` and `val_gene` are reported once, at the selected epoch.
    """
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).unique()
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * 0.15))
    val_groups = set(groups[:n_val])
    rest = groups[n_val:]
    n_inner = max(1, int(len(rest) * inner_val_frac))
    inner_groups = set(rest[:n_inner])

    big_genes = df["target_gene"].value_counts()
    big_genes = big_genes[big_genes >= 300].index.tolist()
    holdout_genes = set(rng.choice(sorted(big_genes), size=n_holdout_genes,
                                   replace=False))

    g = df[group_col].astype(str)
    is_group_val = g.isin(val_groups)
    is_inner_val = g.isin(inner_groups)
    is_gene_hold = df["target_gene"].isin(holdout_genes)
    train = df[~is_group_val & ~is_inner_val & ~is_gene_hold]
    inner_val = df[is_inner_val & ~is_gene_hold]
    val_group = df[is_group_val & ~is_gene_hold]
    val_gene = df[is_gene_hold]
    return train, inner_val, val_group, val_gene, holdout_genes


def report_sequence_overlap(train, parts, seq_col="aso_sequence_5_to_3"):
    """Exact-sequence overlap between train and each held-out slice.

    ASO patent families republish identical sequences and tile a target with
    single-nucleotide shifts, so a group-wise split does not by itself imply
    sequence-level separation. This reports the exact-match component; it does
    NOT detect near-duplicate tiling, which needs an alignment-based check.
    """
    tr_seqs = set(train[seq_col].astype(str))
    out = {}
    for name, part in parts.items():
        s = part[seq_col].astype(str)
        n_ov = int(s.isin(tr_seqs).sum())
        out[name] = dict(n=int(len(s)), n_exact_in_train=n_ov,
                         frac_exact_in_train=round(n_ov / max(1, len(s)), 4))
    return out


def evaluate(model, df_part, gene2row, cell2id, device, dose_mu,
             with_ci=False, ci_seed=0):
    model.eval()
    preds, trues = [], []
    B = 1024
    rows = list(df_part.itertuples())
    with torch.no_grad():
        for lo in range(0, len(rows), B):
            chunk = rows[lo:lo + B]
            base, sug, bb, pad = encode_batch(
                [r.aso_sequence_5_to_3 for r in chunk],
                [r.sugar_seq for r in chunk], [r.backbone_seq for r in chunk])
            gene_rows = torch.tensor([gene2row.get(r.target_gene, gene2row["UNKNOWN"])
                                      for r in chunk], dtype=torch.long)
            cell_ids = torch.tensor([cell2id.get(r.cell_line, cell2id["UNKNOWN"])
                                     for r in chunk], dtype=torch.long)
            dl = np.array([r.dosage_log10 if r.dosage_log10 == r.dosage_log10 else dose_mu
                           for r in chunk], dtype=np.float32)
            dm = np.array([0.0 if r.dosage_log10 == r.dosage_log10 else 1.0
                           for r in chunk], dtype=np.float32)
            out = model(base.to(device), sug.to(device), bb.to(device), pad.to(device),
                        gene_rows.to(device), cell_ids.to(device),
                        torch.from_numpy(dl).to(device), torch.from_numpy(dm).to(device))
            preds.extend(out.cpu().numpy().tolist())
            trues.extend([r.inhibition_clipped for r in chunk])
    preds, trues = np.array(preds), np.array(trues)
    rho = spearmanr(trues, preds).statistic
    r = pearsonr(trues, preds).statistic
    k = max(1, len(trues) // 20)
    top_true = set(np.argsort(-trues)[:k])
    top_pred = set(np.argsort(-preds)[:k])
    # Normalised so that 1.0 == random: E[|intersection|]/k = k/n = 0.05 under a
    # random ranking, hence the factor 20. Releases up to v3 used 5, making the
    # random baseline 0.25 for every previously published enrich_top5 value.
    enrich = len(top_true & top_pred) / k * 20.0
    out = dict(spearman=float(rho), pearson=float(r), enrich_top5=float(enrich),
               enrich_random_baseline=1.0, n=len(trues))
    if with_ci:
        out["spearman_ci95"] = list(_bootstrap_ci(trues, preds, seed=ci_seed))
    return out


def _bootstrap_ci(trues, preds, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(trues)
    if n < 8:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        t, p = trues[idx], preds[idx]
        if np.std(t) < 1e-12 or np.std(p) < 1e-12:
            continue
        v = spearmanr(t, p).statistic
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    print(f"data: {df.shape}", flush=True)

    # ESM2 table (frozen mechanism features)
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab.keys()) if isinstance(tab, dict) else None
    if isinstance(tab, dict):
        esm = torch.stack([v.float() for v in tab.values()])
        gene2row = {s: i for i, s in enumerate(symbols)}
    else:
        esm = tab.float()
        gene2row = {}
    gene2row.setdefault("UNKNOWN", 0)

    df["cell_line"] = df["cell_line"].fillna("UNKNOWN").astype(str)
    cell_lines = sorted(df["cell_line"].unique())
    cell2id = {c: i for i, c in enumerate(cell_lines)}
    cell2id.setdefault("UNKNOWN", len(cell2id))
    dose_mu = float(df["dosage_log10"].dropna().mean())

    train, inner_val, val_group, val_gene, holdout_genes = build_split(
        df, args.seed, args.n_holdout_genes, group_col=args.group_col)
    print(f"train={len(train)} inner-val={len(inner_val)} "
          f"{args.group_col}-group-val={len(val_group)} "
          f"gene-holdout={len(val_gene)} ({len(holdout_genes)} genes: "
          f"{sorted(holdout_genes)[:6]}...)", flush=True)
    overlap = report_sequence_overlap(
        train, {"inner_val": inner_val, "val_group": val_group, "val_gene": val_gene})
    print(f"[leakage-check] exact-sequence overlap with train: "
          f"{json.dumps(overlap)}", flush=True)

    model = EfficacyHead(esm, n_cell_lines=len(cell2id),
                         use_pos_emb=args.use_pos_emb).to(device)
    print(f"params trainable={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(esm table frozen buffer)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    trues_tr = torch.tensor(train["inhibition_clipped"].values, dtype=torch.float32)
    dose_mu_t = torch.tensor(dose_mu)

    # pre-encode train tensors once
    base, sug, bb, pad = encode_batch(train["aso_sequence_5_to_3"].tolist(),
                                      train["sugar_seq"].tolist(),
                                      train["backbone_seq"].tolist())
    gene_rows = torch.tensor([gene2row.get(g, gene2row["UNKNOWN"])
                              for g in train["target_gene"]], dtype=torch.long)
    cell_ids = torch.tensor([cell2id.get(c, cell2id["UNKNOWN"])
                             for c in train["cell_line"]], dtype=torch.long)
    dl = train["dosage_log10"].fillna(dose_mu).values.astype(np.float32)
    dm = train["dosage_log10"].isna().values.astype(np.float32)
    dl_t = torch.from_numpy(dl)
    dm_t = torch.from_numpy(dm)
    # Keep the training tensors on the same device as the model. Previously they
    # stayed on CPU while the model was moved, so any CUDA run raised a
    # device-mismatch error and only the CPU path was ever exercised.
    base, sug, bb, pad = (base.to(device), sug.to(device), bb.to(device), pad.to(device))
    gene_rows, cell_ids = gene_rows.to(device), cell_ids.to(device)
    dl_t, dm_t, trues_tr = dl_t.to(device), dm_t.to(device), trues_tr.to(device)

    log_path = Path(args.out_dir) / "train_log.jsonl"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "ckpt_efficacy_v1.pt"
    best, best_ep = -np.inf, -1
    n = len(train)
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            out = model(base[idx], sug[idx], bb[idx], pad[idx],
                        gene_rows[idx], cell_ids[idx], dl_t[idx], dm_t[idx])
            loss = nn.functional.huber_loss(out, trues_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(loss.item()))
        sched.step()
        # Model selection uses ONLY the inner validation groups. val_group and
        # val_gene are reported once, after training, at the selected epoch --
        # selecting on a slice and then reporting that same slice is what made
        # the previously published v1 "val spearman 0.50" optimistically biased.
        ev_inner = evaluate(model, inner_val, gene2row, cell2id, device, dose_mu)
        msg = dict(epoch=epoch + 1, train_loss=float(np.mean(run)),
                   secs=round(time.time() - t0, 1), inner_val=ev_inner)
        print(json.dumps(msg), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        if ev_inner["spearman"] > best:
            best, best_ep = ev_inner["spearman"], epoch + 1
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "arch_config": model.arch_config(),
                        "selection_metric": "inner_val_spearman",
                        "selection_value": float(best),
                        "cell2id": cell2id, "gene2row": gene2row,
                        "dose_mu": dose_mu, "group_col": args.group_col},
                       ckpt_path)

    # ---- single, final evaluation of both reported slices at the selected epoch ----
    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    ev_group = evaluate(model, val_group, gene2row, cell2id, device, dose_mu,
                        with_ci=True, ci_seed=args.seed)
    ev_gene = evaluate(model, val_gene, gene2row, cell2id, device, dose_mu,
                       with_ci=True, ci_seed=args.seed + 1)
    final = dict(
        selected_epoch=best_ep, selection="inner_val_spearman", selection_value=float(best),
        group_col=args.group_col,
        group_holdout=ev_group,
        gene_holdout=ev_gene,
        holdout_genes=sorted(holdout_genes),
        sequence_overlap_with_train=overlap,
        note=("enrich_top5 is normalised so 1.0 == random; a gene-holdout value "
              "near 1.0 means top-5% selection on unseen target genes is no better "
              "than chance. group_col is the ASO Atlas source patent TABLE unless "
              "an explicit patent column was passed, so this is not a patent split."),
    )
    print(f"[FINAL ep{best_ep}] {json.dumps(final, indent=2)}", flush=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"final": final}) + "\n")
    with open(Path(args.out_dir) / "final_report.json", "w") as f:
        json.dump(final, f, indent=2)
    # Re-save the checkpoint with the reported metrics attached, so that anything
    # loading it (e.g. predict_service) advertises held-out numbers rather than
    # the selection slice's.
    ck["metrics"] = dict(group_holdout=ev_group, gene_holdout=ev_gene)
    torch.save(ck, ckpt_path)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
