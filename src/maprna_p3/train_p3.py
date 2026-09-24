"""VCPE-rna P2.3-B training: direct per-gene deviation head (no SE backbone).

Protocol (inherited from P2.1/P2.2, kept intact):
  true_fc      = pert_mean_hvg - dataset_ctrl_mean_hvg
  common_fc    = mean over TRAIN perturbations only (no leakage)
  true_dev     = true_fc - common_fc   <- the model predicts THIS directly
  pred_fc      = pred_dev + common_fc  (for full-fc metrics)

Eval reports full-fc metrics (mse_DE / pearson_delta / top50) AND dev metrics
(mse_dev / pearson_dev / top50_dev) PLUS a built-in conditioning ablation:
each epoch, pred_dev is recomputed with the target ESM2 vector zeroed and
shuffled; ablation_r = corr(real, zero). P2/P2.1/P2.2 all showed 1.000 (dead).
If conditioning is alive, ablation_r drops well below 1.

No SE / flash-attention / MAP repo / env vars required. Runs in minutes.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
for _sib in ("maprna_p1", "maprna_p2", "."):
    _p = _HERE.parent / _sib
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ds_knockdown import build_symbol2row, load_kd_datasets, make_hvg_list, resolve_pert_row  # noqa: E402
from model_dev import DeviationModel, load_esm_matrix, load_dev_state  # noqa: E402
from baselines import run_all  # noqa: E402
from eval_metrics import (  # noqa: E402
    per_item_correlation, pooled_correlation, top_k_overlap)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--string_neighbors", type=str, default="",
                   help="string_neighbors.npy (network axis); empty = off")
    p.add_argument("--rna_encoder_ckpt", type=str, default="",
                   help="aligned RNA encoder (.pt, Stage A); empty = no seq axis")
    p.add_argument("--fasta", type=str, default="",
                   help="gene_transcripts.fa (required if rna_encoder_ckpt set)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--split_by", type=str, default="target_gene",
                   choices=["target_gene", "pert"],
                   help="target_gene (default): hold out whole target genes, so a "
                        "test perturbation is genuinely unseen. pert: the pre-v4 "
                        "(dataset, condition) split, which lets the same gene appear "
                        "in both splits via the genome-wide screen -- kept only for "
                        "reproducing historical numbers.")
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inner_val_frac", type=float, default=0.15,
                   help="fraction of TRAINING items held out for epoch/checkpoint "
                        "selection; the test split is scored once, afterwards")
    p.add_argument("--knn_k", type=int, default=10,
                   help="neighbours for the ESM2 retrieval control baseline")
    p.add_argument("--d_model", type=int, default=256)
    return p.parse_args()


# ---------------- data ----------------

def build_dev_data(args):
    sym2row, esm_dim = build_symbol2row(args.esm_table)
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)
    hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg)

    # dataset-level control mean over HVG (deterministic ctrl baseline)
    ctrl_mean = np.zeros((len(kd["X_ctrl"]), len(hvg_rows)), dtype=np.float32)
    for di, (Xc, rows) in enumerate(zip(kd["X_ctrl"], kd["row_of_gene"])):
        cm = Xc.mean(axis=0)
        row2col = {int(r): c for c, r in enumerate(rows)}
        for hi, hr in enumerate(hvg_rows):
            c = row2col.get(int(hr))
            if c is not None:
                ctrl_mean[di, hi] = cm[c]
    # z-score across genes within each dataset (feature scale for the MLP)
    mu, sd = ctrl_mean.mean(axis=1, keepdims=True), ctrl_mean.std(axis=1, keepdims=True) + 1e-6
    ctrl_feat_all = (ctrl_mean - mu) / sd

    # perturbation items: all (ds, condition) with enough cells and a resolvable target
    items = []
    for di in range(len(kd["X_pert"])):
        cp = kd["pert_labels"][di]
        for cond in sorted(set(cp)):
            if cond.lower() == "ctrl":
                continue
            idx = np.where(cp == cond)[0]
            # after pseudobulk aggregation each condition has exactly 1 row;
            # the >=min_cells gate already ran on cell counts inside load_kd_datasets
            if len(idx) < 1:
                continue
            if resolve_pert_row(sym2row, cond) < 0:
                continue
            items.append((di, cond))
    rng = np.random.default_rng(args.seed)
    items = sorted(items)
    if args.split_by == "target_gene":
        # Group by the ESM2 row the condition resolves to, i.e. by target gene,
        # so that no target gene appears on both sides of the split.
        #
        # Why this matters: the expanded training set includes Replogle gwps, a
        # genome-wide K562 screen that covers essentially every target gene in
        # the legacy adamson/norman/replogle_ess datasets. Splitting on
        # (dataset, condition) pairs -- the pre-v4 behaviour, still available as
        # --split_by pert -- therefore lets the SAME target gene sit in train
        # (via gwps) and in test (via a legacy dataset). Since a perturbation is
        # represented to the model only by (pert_row, rna_emb), such a test item
        # is not unseen. Any "unseen perturbation" claim requires split_by
        # target_gene.
        by_gene = {}
        for di, cond in items:
            by_gene.setdefault(resolve_pert_row(sym2row, cond), []).append((di, cond))
        genes = sorted(by_gene)
        rng.shuffle(genes)
        n_test_g = max(1, int(round(len(genes) * args.test_frac)))
        test_genes = set(genes[:n_test_g])
        test_items = sorted(it for g in test_genes for it in by_gene[g])
        train_items = sorted(it for g in genes[n_test_g:] for it in by_gene[g])
        print(f"[split] by target_gene: {len(genes)} genes -> "
              f"train {len(train_items)} items / test {len(test_items)} items "
              f"({n_test_g} held-out genes)", flush=True)
    else:
        rng.shuffle(items)
        n_test = max(1, int(round(len(items) * args.test_frac)))
        test_items, train_items = sorted(items[:n_test]), sorted(items[n_test:])
        overlap = (set(resolve_pert_row(sym2row, c) for _, c in train_items)
                   & set(resolve_pert_row(sym2row, c) for _, c in test_items))
        print(f"[split] by (dataset, condition): train {len(train_items)} / "
              f"test {len(test_items)} | WARNING: {len(overlap)} target genes "
              f"appear in BOTH splits -- results are not an unseen-gene estimate; "
              f"use --split_by target_gene for that.", flush=True)

    # targets: deterministic pert-mean over HVG.
    # `measured` marks HVG columns actually present in that item's dataset panel.
    # Unmeasured columns stay 0, which makes fc == 0 and dev == -common_fc,
    # a value that is constant across every perturbation of the dataset and
    # carries no perturbation-specific signal. Pre-v4 those columns entered the
    # loss and every metric unmasked, so a model could score on them using
    # ds_emb + ctrl_feat alone -- the same shared-component inflation the P2
    # erratum documented, one level down.
    def targets_of(split_items):
        T = np.zeros((len(split_items), len(hvg_rows)), dtype=np.float32)
        M = np.zeros((len(split_items), len(hvg_rows)), dtype=bool)
        rows_p = np.zeros(len(split_items), dtype=np.int64)
        for k, (di, cond) in enumerate(split_items):
            cp = kd["pert_labels"][di]
            idx = np.where(cp == cond)[0]
            mean = kd["X_pert"][di][idx].mean(axis=0)
            row2col = {int(r): c for c, r in enumerate(kd["row_of_gene"][di])}
            for hi, hr in enumerate(hvg_rows):
                c = row2col.get(int(hr))
                if c is not None:
                    T[k, hi] = mean[c]
                    M[k, hi] = True
            rows_p[k] = resolve_pert_row(sym2row, cond)
        return T, rows_p, M

    T_tr, rows_tr, mask_tr = targets_of(train_items)
    T_te, rows_te, mask_te = targets_of(test_items)
    print(f"[mask] measured HVG fraction: train {mask_tr.mean():.3f} "
          f"test {mask_te.mean():.3f} (unmeasured columns are excluded from "
          f"loss and metrics)", flush=True)

    # V2-1a: 靶基因自身 ctrl 表达（raw log1p，供 self-response 门控）。
    # 目标基因在所属数据集 panel 内 -> 取该列 ctrl 均值；panel 外 -> 0.0
    #（此时目标一般也不在 HVG 响应 panel，is_tgt 行不存在，门控不起作用，安全）。
    def pert_expr_of(split_items):
        E = np.zeros(len(split_items), dtype=np.float32)
        for k, (di, cond) in enumerate(split_items):
            r = resolve_pert_row(sym2row, cond)
            if r < 0:
                continue
            row2col = {int(rr): c for c, rr in enumerate(kd["row_of_gene"][di])}
            c = row2col.get(int(r))
            if c is not None:
                E[k] = float(kd["X_ctrl"][di].mean(axis=0)[c])
        return E

    pert_expr_tr = pert_expr_of(train_items)
    pert_expr_te = pert_expr_of(test_items)
    print(f"[v2-1a] target ctrl expr: train median {np.median(pert_expr_tr):.3f} "
          f"| test median {np.median(pert_expr_te):.3f} "
          f"| near-zero(<0.1) train {int((pert_expr_tr < 0.1).sum())}/{len(pert_expr_tr)}",
          flush=True)

    # full fc and residual targets (train-only common core).
    # The common core is a MASKED mean: averaging raw zeros from unmeasured
    # columns would shrink it toward zero on exactly those genes that are
    # panel-specific, distorting the residual everywhere downstream.
    fc_tr = T_tr - ctrl_mean[np.array([di for di, _ in train_items])]
    fc_te = T_te - ctrl_mean[np.array([di for di, _ in test_items])]
    denom = mask_tr.sum(axis=0)
    common_fc = np.where(denom > 0, (fc_tr * mask_tr).sum(axis=0) / np.maximum(denom, 1), 0.0)
    common_fc = common_fc.astype(np.float32)
    dev_tr, dev_te = fc_tr - common_fc, fc_te - common_fc
    n_never = int((denom == 0).sum())
    print(f"[P2.3-B] common fc (train-only, masked): std={common_fc.std():.4f} "
          f"max|.|={np.abs(common_fc).max():.4f} | HVG columns never measured in "
          f"train: {n_never}/{len(hvg_rows)}", flush=True)

    # Inner validation split for epoch/checkpoint selection, carved out of the
    # TRAINING items and grouped the same way as the outer split. Selecting the
    # checkpoint on the test split -- the pre-v4 behaviour -- turns the reported
    # pearson_dev into a maximum over epochs rather than an estimate.
    rng_in = np.random.default_rng(args.seed + 977)
    if args.split_by == "target_gene":
        tr_genes = sorted({resolve_pert_row(sym2row, c) for _, c in train_items})
        rng_in.shuffle(tr_genes)
        n_in = max(1, int(round(len(tr_genes) * args.inner_val_frac)))
        val_genes = set(tr_genes[:n_in])
        is_inner_val = np.array([resolve_pert_row(sym2row, c) in val_genes
                                 for _, c in train_items])
    else:
        perm = rng_in.permutation(len(train_items))
        n_in = max(1, int(round(len(train_items) * args.inner_val_frac)))
        is_inner_val = np.zeros(len(train_items), dtype=bool)
        is_inner_val[perm[:n_in]] = True
    print(f"[split] inner-val for selection: {int(is_inner_val.sum())} of "
          f"{len(train_items)} training items (test split is scored once, at the "
          f"selected epoch)", flush=True)

    return dict(sym2row=sym2row, esm_dim=esm_dim, hvg_rows=hvg_rows,
                ctrl_feat_all=ctrl_feat_all, train_items=train_items,
                test_items=test_items, rows_tr=rows_tr, rows_te=rows_te,
                pert_expr_tr=pert_expr_tr, pert_expr_te=pert_expr_te,
                fc_tr=fc_tr, fc_te=fc_te, dev_tr=dev_tr, dev_te=dev_te,
                mask_tr=mask_tr, mask_te=mask_te, is_inner_val=is_inner_val,
                common_fc=common_fc, n_ds=len(kd["X_ctrl"]))


def make_rna_emb_lookup(args, data):
    """Precompute RNA-encoder embeddings for all perturbations (train+test)."""
    if not args.rna_encoder_ckpt:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maprna_p2"))
    from rna_encoder import RNAEncoder, encode_seq, load_fasta_symbol_seqs, lookup_seq
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = RNAEncoder(d_model=256, max_len=600).to(dev).eval()
    state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "rna_encoder" in state:
        state = state["rna_encoder"]        # Stage-A ckpt is a wrapper dict
    enc.load_state_dict(state)
    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    all_items = data["train_items"] + data["test_items"]
    embs = np.zeros((len(all_items), 256), dtype=np.float32)
    with torch.no_grad():
        for k, (di, cond) in enumerate(all_items):
            gene = cond.split("+")[0]
            seq = lookup_seq(gene, symbol_map, ensembl_map)
            if seq:
                ids, m = encode_seq(seq, 600)
            else:
                ids, m = [6] + [0] * 599, [1] + [0] * 599  # CLS + PAD fallback
            tok = torch.tensor([ids], device=dev)
            msk = torch.tensor([m], device=dev).bool()
            embs[k] = enc(tok, msk)[0].float().cpu().numpy()
    print(f"[P2.3-B] rna embeddings: {len(all_items)} perts", flush=True)
    return torch.from_numpy(embs)


# ---------------- eval ----------------

@torch.no_grad()
def evaluate(model, data, split, device, esm_override_mode=None):
    """split: 'test' | 'train' | 'inner_val' | 'inner_train'.

    'inner_val' is the slice used for epoch/checkpoint selection; 'test' must be
    scored once, after selection. Pre-v4 the checkpoint was saved on the best
    'test' pearson_dev, making the reported value a maximum over epochs.
    """
    is_test = split == "test"
    rows = data["rows_te"] if is_test else data["rows_tr"]
    fc = data["fc_te"] if is_test else data["fc_tr"]
    dev = data["dev_te"] if is_test else data["dev_tr"]
    items = data["test_items"] if is_test else data["train_items"]
    mask_all = data["mask_te"] if is_test else data["mask_tr"]
    pe_all = data["pert_expr_te"] if is_test else data["pert_expr_tr"]
    rna_all = None
    if model.proj_r is not None:
        rna_all = data["rna_embs"][len(data["train_items"]):] if is_test \
            else data["rna_embs"][:len(data["train_items"])]

    if split in ("inner_val", "inner_train"):
        sel = data["is_inner_val"]
        if split == "inner_train":
            sel = ~sel
        keep = np.flatnonzero(sel)
        rows, fc, dev = rows[keep], fc[keep], dev[keep]
        mask_all, pe_all = mask_all[keep], pe_all[keep]
        items = [items[i] for i in keep]
        if rna_all is not None:
            rna_all = rna_all[torch.from_numpy(keep)]

    ds_idx = torch.tensor([di for di, _ in items], dtype=torch.long)
    pr = torch.tensor(rows, dtype=torch.long)
    cf = torch.from_numpy(data["ctrl_feat_all"][ds_idx.numpy()])
    pe = torch.from_numpy(pe_all)
    rna = rna_all

    def _predict_chunks(pr, ds_idx, cf, rna, pe, ov=None, chunk=128):
        outs = []
        for lo in range(0, pr.shape[0], chunk):
            hi = lo + chunk
            kw = {"pert_esm_override": ov[lo:hi]} if ov is not None else {}
            outs.append(model(pr[lo:hi], ds_idx[lo:hi], cf[lo:hi],
                              rna_emb=rna[lo:hi] if rna is not None else None,
                              pert_ctrl_expr=pe[lo:hi], **kw).cpu())
        return torch.cat(outs).numpy()

    pred_dev = _predict_chunks(pr, ds_idx, cf, rna, pe)
    if esm_override_mode == "zero":
        ov = torch.zeros_like(model.esm_table[pr])
        pred_dev = _predict_chunks(pr, ds_idx, cf, rna, pe, ov=ov)
    elif esm_override_mode == "shuffle":
        g = torch.Generator().manual_seed(123)
        perm = torch.randperm(pr.shape[0], generator=g)
        ov = model.esm_table[pr[perm]]
        pred_dev = _predict_chunks(pr, ds_idx, cf, rna, pe, ov=ov)

    # Restrict every metric to HVG columns actually measured in that item's
    # dataset panel. Unmeasured columns have dev == -common_fc identically for
    # every perturbation of the dataset, so scoring them rewards predicting a
    # dataset-level constant and inflates the per-perturbation correlations.
    mask = mask_all

    pred_fc = pred_dev + data["common_fc"]
    out = {}
    out["measured_fraction"] = float(mask.mean())

    def _masked_mse(pred, true):
        d2 = ((pred - true) ** 2) * mask
        denom = np.maximum(mask.sum(axis=1), 1)
        return float((d2.sum(axis=1) / denom).mean())

    def _per_item(pred, true, k=50):
        """Per-perturbation Pearson and top-k overlap over measured columns only."""
        prs, tops = [], []
        for b in range(len(true)):
            m = mask[b]
            if m.sum() < 10:
                continue
            t, p = true[b][m], pred[b][m]
            if t.std() > 1e-6 and p.std() > 1e-6:
                prs.append(float(np.corrcoef(t, p)[0, 1]))
            else:
                prs.append(0.0)
            kk = min(k, len(t))
            tops.append(len(set(np.argsort(-np.abs(t))[:kk])
                            & set(np.argsort(-np.abs(p))[:kk])) / kk)
        if not prs:
            return float("nan"), float("nan"), 0
        return float(np.mean(prs)), float(np.mean(tops)), len(prs)

    out["mse_DE"] = _masked_mse(pred_fc, fc)
    out["mse_DE_baseline_ctrl"] = _masked_mse(np.zeros_like(fc), fc)
    out["pearson_delta"], out["top50_deg_overlap"], _ = _per_item(pred_fc, fc)
    # dev metrics
    out["mse_dev"] = _masked_mse(pred_dev, dev)
    pd_, td_, n_scored = _per_item(pred_dev, dev)
    out["pearson_dev"], out["top50_dev"] = pd_, td_
    out["n_scored_perturbations"] = n_scored
    # Estimator label, so that this number is never silently compared against
    # eval_fair.py's flattened pooled correlation -- they are different
    # estimators and differ by ~1.5x on the same checkpoint.
    out["pearson_dev_estimator"] = "mean_over_perturbations_of_within_perturbation_r"
    return out


# ---------------- main ----------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = build_dev_data(args)
    esm = load_esm_matrix(args.esm_table)

    rna_enc = None
    if args.rna_encoder_ckpt:
        from rna_encoder import RNAEncoder
        rna_enc = RNAEncoder(d_model=256, max_len=600)
        state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "rna_encoder" in state:
            state = state["rna_encoder"]    # Stage-A ckpt is a wrapper dict
        rna_enc.load_state_dict(state)
        data["rna_embs"] = make_rna_emb_lookup(args, data)

    model = DeviationModel(esm, data["hvg_rows"], n_ds=data["n_ds"],
                           d_model=args.d_model, rna_encoder=rna_enc).to(device)
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
        cov = int((model.neighbor_table >= 0).any(dim=1).sum())
        print(f"[P2.3-B] neighbor table: {tuple(model.neighbor_table.shape)} "
              f"(coverage {cov}/{model.neighbor_table.shape[0]})", flush=True)
    else:
        print("[P2.3-B] network axis OFF", flush=True)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"params total={n_par/1e6:.1f}M (all trainable)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dev_tr_t = torch.from_numpy(data["dev_tr"]).float().to(device)
    # Same masking rationale as in evaluate(): unmeasured HVG columns carry a
    # dataset-level constant target and no perturbation-specific signal, so
    # training on them teaches the model to reproduce that constant.
    mask_tr_t = torch.from_numpy(data["mask_tr"]).float().to(device)
    ds_tr = torch.tensor([di for di, _ in data["train_items"]], dtype=torch.long)
    rows_tr = torch.tensor(data["rows_tr"], dtype=torch.long)
    cf_tr = torch.from_numpy(data["ctrl_feat_all"][ds_tr.numpy()]).float().to(device)
    pe_tr = torch.from_numpy(data["pert_expr_tr"]).float().to(device)
    rna_tr = (data["rna_embs"][:len(data["train_items"])].to(device)
              if model.proj_r is not None else None)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    best_pd, best_epoch = -float('inf'), -1
    ckpt_path = os.path.join(args.out_dir, 'ckpt_p3_best_dev.pt')
    n = len(rows_tr)
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        perm = torch.randperm(n)
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            pred = model(rows_tr[idx], ds_tr[idx], cf_tr[idx],
                         rna_emb=rna_tr[idx] if rna_tr is not None else None,
                         pert_ctrl_expr=pe_tr[idx])
            m = mask_tr_t[idx]
            loss = (((pred - dev_tr_t[idx]) ** 2) * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(loss.item()))
        sched.step()
        msg = dict(epoch=epoch + 1, steps=(n + args.batch_size - 1) // args.batch_size,
                   train_loss=float(np.mean(run)), secs=round(time.time() - t0, 1))
        model.eval()
        with torch.no_grad():
            ev_in = evaluate(model, data, "inner_val", device)
            ev = ev_in  # logged per epoch; the test split is NOT scored here
        # ablation r on the INNER-VAL slice: the test split is reserved for a
        # single scoring pass after selection.
        with torch.no_grad():
            _sel = np.flatnonzero(data["is_inner_val"])
            rows_t = torch.tensor(data["rows_tr"][_sel], dtype=torch.long)
            ds_t = torch.tensor([data["train_items"][i][0] for i in _sel], dtype=torch.long)
            cf_t = torch.from_numpy(data["ctrl_feat_all"][ds_t.numpy()])
            pe_t = torch.from_numpy(data["pert_expr_tr"][_sel])
            rna_t = (data["rna_embs"][:len(data["train_items"])][torch.from_numpy(_sel)]
                     if model.proj_r is not None else None)
            ov_all = torch.zeros_like(model.esm_table[rows_t])

            def _abl_chunks(chunk=128):
                """Single-factor ablation: ONLY the target-gene ESM2 vector changes.

                Pre-v4 this passed `rna_emb` to the ablated branch but not to the
                real one, so it compared "no RNA axis + true ESM2" against
                "RNA axis + zeroed ESM2" -- two factors at once, and neither
                branch matched the configuration used by evaluate(). Every
                ablation_r logged by a run with the RNA axis enabled (including
                results/p3_v21e/train_log.jsonl) is affected.

                Note the remaining caveat: zeroing the ESM2 vector does not
                remove perturbation identity, because is_target / is_neighbor
                still encode it. Use --ablation_mode to isolate those axes.
                """
                reals, zeros = [], []
                for lo in range(0, rows_t.shape[0], chunk):
                    hi = lo + chunk
                    rna_chunk = rna_t[lo:hi] if rna_t is not None else None
                    reals.append(model(rows_t[lo:hi], ds_t[lo:hi], cf_t[lo:hi],
                                       rna_emb=rna_chunk,
                                       pert_ctrl_expr=pe_t[lo:hi]).cpu())
                    zeros.append(model(rows_t[lo:hi], ds_t[lo:hi], cf_t[lo:hi],
                                       rna_emb=rna_chunk,
                                       pert_esm_override=ov_all[lo:hi],
                                       pert_ctrl_expr=pe_t[lo:hi]).cpu())
                return torch.cat(reals).numpy().ravel(), torch.cat(zeros).numpy().ravel()

            p_real, p_zero = _abl_chunks()
        ab_r = float(np.corrcoef(p_real, p_zero)[0, 1]) if p_real.std() > 1e-9 else 1.0
        msg["eval"] = ev
        msg["ablation_r_real_vs_zeropert"] = round(ab_r, 4)
        print(json.dumps(msg), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        if ev_in["pearson_dev"] > best_pd:
            best_pd, best_epoch = ev_in["pearson_dev"], epoch + 1
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "selection": "inner_val_pearson_dev",
                        "selection_value": float(best_pd),
                        "hvg_rows": data["hvg_rows"],
                        "common_fc": data["common_fc"],
                        "split_by": args.split_by},
                       ckpt_path)
            print(f"saved best inner-val ckpt (pearson_dev={best_pd:.4f}, "
                  f"ep{epoch + 1})", flush=True)

    # ---- single, final scoring of the held-out split, at the selected epoch ----
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_dev_state(model, ck)
    model.to(device).eval()
    final = evaluate(model, data, "test", device)
    final["selected_epoch"] = best_epoch
    final["inner_val_pearson_dev_at_selection"] = float(best_pd)
    final["split_by"] = args.split_by

    # ---- control baselines on the same split, same mask, same metrics ----
    # Without these there is no evidence that the learned head does more than
    # retrieval over the frozen ESM2 embeddings it is conditioned on.
    esm_np = model.esm_table.detach().cpu().numpy()
    base_preds = run_all(
        data["dev_tr"], esm_np[data["rows_tr"]], esm_np[data["rows_te"]],
        mask_tr=data["mask_tr"], knn_k=args.knn_k)
    mask_te, dev_te = data["mask_te"], data["dev_te"]
    final["baselines"] = {}
    for name, pred in base_preds.items():
        final["baselines"][name] = dict(
            pearson_dev=per_item_correlation(dev_te, pred, mask_te),
            pearson_dev_pooled=pooled_correlation(dev_te, pred, mask_te),
            top50_dev=top_k_overlap(dev_te, pred, k=50, mask=mask_te))

    print("\n=== held-out results (scored once, at the selected epoch) ===",
          flush=True)
    print(f"{'model':<14} pearson_dev={final['pearson_dev']:.4f}  "
          f"top50_dev={final['top50_dev']:.4f}", flush=True)
    for name, m in final["baselines"].items():
        print(f"{name:<14} pearson_dev={m['pearson_dev']:.4f}  "
              f"top50_dev={m['top50_dev']:.4f}", flush=True)
    beaten = [n for n, m in final["baselines"].items()
              if final["pearson_dev"] <= m["pearson_dev"]]
    if beaten:
        print(f"\nNOTE: the trained head does NOT beat {beaten} on pearson_dev. "
              f"A conditioned model that loses to retrieval or to a linear map "
              f"over the same embeddings has not been shown to learn "
              f"perturbation-specific biology.", flush=True)

    ck["metrics"] = final
    torch.save(ck, ckpt_path)
    with open(os.path.join(args.out_dir, "final_report.json"), "w") as f:
        json.dump(final, f, indent=2)
    with open(log_path, "a") as f:
        f.write(json.dumps({"final": final}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
