#!/usr/bin/env python
"""V2-2 re-opened (Route A): set-encoder cell-state embedding (panel-agnostic).

The fundamental difference from the v1 version (already FAILed and shut down):
v1 built the profile vector from the "intersection of the 6 context panels"
(312 genes) -- the intersection was so small that both zero-shot and fallback
failed. Route A switches to a (gene identity, expression value) set encoding:
every context contributes the genes it measured (ESM2 emb ⊕ z-score) →
attention pooling → a 64d embedding. A missing gene simply does not contribute,
so panel differences are no longer a constraint. The response panel goes back to
the production convention: make_hvg_list, 2000 HVG across datasets (the same as
train_p3).

Re-pre-registered criterion (hepg2 holdout; FAIL = permanently closed, there is
no second re-opening):
  zero_shot pearson_dev - fallback pearson_dev >= +0.02 and zero_shot > 0.15
  the fallback baseline = copying verbatim the discrete ds_emb of the nearest
  training context as judged by the encoder.

Usage (GPU box, flat $BASE/src/ layout; compatible with the local
maprna_p1/p3 sibling directories):
  python train_cell_embedding.py \
      --data_dirs adamson/perturb_processed.h5ad norman/perturb_processed.h5ad \
                 replogle_rpe1_essential/perturb_processed.h5ad \
                 scperturb/ReplogleWeissman2022_K562_gwps_proc.h5ad \
                 scperturb/NadigOConner2024_jurkat_proc.h5ad \
                 scperturb/NadigOConner2024_hepg2_proc.h5ad \
      --context_names k562a norman rpe1 k562g jurkat hepg2 \
      --esm_table Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
      --holdout hepg2 --out_dir $BASE/cell_emb_v2_hepg2
Smoke test: --epochs 2 --max_items 300
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
_HERE = os.path.abspath(HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# Local repo layout: maprna_p1/p2/p3 sibling directories (with the flat src/
# layout on the GPU box the is_dir guard skips them)
for _sib in ("maprna_p1", "maprna_p2", "maprna_p3"):
    _p = os.path.abspath(os.path.join(HERE, "..", _sib))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from ds_knockdown import (  # noqa: E402
    load_kd_datasets, build_symbol2row, resolve_pert_row, make_hvg_list)
from model_dev import DeviationModel, load_esm_matrix, GATE_TAU, GATE_SLOPE  # noqa: E402

DS_EMB = 8
D_CELL = 64
D_MODEL = 256


class CellStateEncoder(nn.Module):
    """Route A set-encoder: a gene set of (ESM2 emb, z-score expr) -> attention
    pooling -> d_cell.

    Panel-agnostic: every context contributes only the genes it measured;
    missing genes do not take part in the pooling.
    """

    def __init__(self, esm_dim, d_model=D_MODEL, d_cell=D_CELL):
        super().__init__()
        self.proj_esm = nn.Linear(esm_dim, d_model)
        self.inp = nn.Linear(d_model + 1, d_model)            # [esm_proj; z]
        self.attn_v = nn.Linear(d_model, 1, bias=False)
        self.out = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, d_cell), nn.GELU())

    def forward(self, esm_table, rows_t, z_t):
        """esm_table [V, esm_dim]; rows_t [n_panel] long; z_t [n_panel] float."""
        e = self.proj_esm(esm_table[rows_t])                  # [n_panel, d_model]
        h = self.inp(torch.cat([e, z_t.unsqueeze(-1)], dim=-1))
        a = torch.softmax(self.attn_v(torch.tanh(h)), dim=0)  # [n_panel, 1]
        return self.out((a * h).sum(dim=0))                   # [d_cell]


class CellEmbDeviationModel(DeviationModel):
    """DeviationModel subclass in which cell_emb replaces the ds one-hot
    (minimally invasive).
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ds_proj = nn.Linear(D_CELL, DS_EMB)              # d_cell -> DS_EMB slot

    def forward(self, pert_rows, ds_idx, ctrl_feat, rna_emb=None,
                pert_esm_override=None, pert_ctrl_expr=None, cell_emb=None):
        Bp = pert_rows.shape[0]
        dev = self.esm_table.device
        if cell_emb is not None:
            ds_onehot = self.ds_proj(cell_emb.to(dev))        # [Bp, DS_EMB]
        else:  # fallback baseline: verbatim discrete ds_emb of the nearest
               # training context
            ds_onehot = self.ds_emb(ds_idx.to(dev))

        ctrl_feat = ctrl_feat.to(dev)
        G = self.proj_g(self.esm_table[self.hvg_rows])
        p_esm = (self.esm_table[pert_rows.to(dev)] if pert_esm_override is None
                 else pert_esm_override.to(dev))
        P = self.proj_p(p_esm)
        inter = G.unsqueeze(0) * P.unsqueeze(1)
        parts = [G.unsqueeze(0).expand(Bp, -1, -1),
                 P.unsqueeze(1).expand(-1, self.n_hvg, -1), inter]
        if self.proj_r is not None and rna_emb is not None:
            R = self.proj_r(rna_emb.to(dev))
            parts.append(R.unsqueeze(1).expand(-1, self.n_hvg, -1))
        else:
            parts.append(torch.zeros(Bp, self.n_hvg, G.shape[-1], device=dev))
        nb = (self.neighbor_table[pert_rows.to(dev)]
              if getattr(self, "neighbor_table", None) is not None
              else torch.full((Bp, 0), -1, device=dev, dtype=torch.long))
        if nb.shape[1] > 0:
            is_nb = (nb.unsqueeze(-1) == self.hvg_rows.view(1, 1, -1)).any(dim=1)
        else:
            is_nb = torch.zeros(Bp, self.n_hvg, dtype=torch.bool, device=dev)
        is_tgt = (pert_rows.to(dev).unsqueeze(-1) == self.hvg_rows.view(1, -1))
        feats = [ctrl_feat.unsqueeze(-1), is_nb.float().unsqueeze(-1),
                 is_tgt.float().unsqueeze(-1),
                 ds_onehot.unsqueeze(1).expand(-1, self.n_hvg, -1)]
        x = torch.cat(parts + feats, dim=-1)
        out = self.mlp(x).squeeze(-1)
        if pert_ctrl_expr is not None:
            gate = torch.sigmoid((pert_ctrl_expr.to(dev) - GATE_TAU) * GATE_SLOPE)
            out = out * (1.0 - is_tgt.float() * (1.0 - gate).unsqueeze(-1))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--context_names", nargs="+", required=True)
    ap.add_argument("--esm_table", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--holdout", default="",
                    help="name of the held-out context; empty = train on all "
                         "contexts")
    ap.add_argument("--n_hvg", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_cells", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_items", type=int, default=0,
                    help="for smoke tests: truncate the number of training "
                         "items (0=all)")
    args = ap.parse_args()
    assert len(args.data_dirs) == len(args.context_names)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    # ---- load every context in one go (production semantics) ----
    sym2row, _ = build_symbol2row(args.esm_table)
    sym_by_row = {int(r): s for s, r in sym2row.items()}
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)
    n_ds = len(kd["X_ctrl"])
    assert n_ds == len(args.context_names)

    # per context: panel rows (ESM rows), sym2col, ctrl mean, z-score, item list
    ctx = []
    for di in range(n_ds):
        rows = [int(r) for r in kd["row_of_gene"][di]]
        sym2col = {}
        for i, r in enumerate(rows):
            s = sym_by_row.get(r)
            if s is not None:
                sym2col[s] = i
        # rows/z/sym2col must all be on the same convention (restricted to the
        # panel subset covered by the ESM table)
        panel_syms = list(sym2col.keys())
        rows = [sym2row[s] for s in panel_syms]
        cm = np.asarray(kd["X_ctrl"][di].mean(axis=0), dtype=np.float32)
        v = cm[[sym2col[s] for s in sym2col]]                 # in-panel values
                                                              # (column-aligned)
        z = (v - v.mean()) / (v.std() + 1e-6)
        cp = kd["pert_labels"][di]
        items = []
        for cond in sorted(set(cp)):
            if cond.lower() == "ctrl":
                continue
            if resolve_pert_row(sym2row, cond) < 0:
                continue
            idx = np.where(cp == cond)[0]
            if len(idx) < 1:
                continue
            mean = np.asarray(kd["X_pert"][di][idx].mean(axis=0), dtype=np.float32)
            items.append((cond, mean))
        ctx.append(dict(name=args.context_names[di], rows=rows, sym2col=sym2col,
                        cm=cm, z=z, items=items))
        print(f"[ctx] {args.context_names[di]}: {len(items)} perts | panel {len(rows)}",
              flush=True)

    # ---- response panel: production convention (HVG across datasets,
    # unmeasured genes recorded as 0, same as train_p3) ----
    hvg_rows = [int(r) for r in make_hvg_list(kd, n_hvg=args.n_hvg)]
    hvg_col_per_ds = []
    for di in range(n_ds):
        r2c = {int(r): c for c, r in enumerate(kd["row_of_gene"][di])}
        hvg_col_per_ds.append(np.array([r2c.get(hr, -1) for hr in hvg_rows],
                                       dtype=np.int64))
    print(f"[data] response panel: {len(hvg_rows)} HVG (across datasets)", flush=True)

    # ---- ctrl features (same as train_p3: within-dataset z-score) ----
    ctrl_mean = np.zeros((n_ds, len(hvg_rows)), dtype=np.float32)
    for di in range(n_ds):
        cm = kd["X_ctrl"][di].mean(axis=0)
        for hi, hr in enumerate(hvg_rows):
            c = hvg_col_per_ds[di][hi]
            if c >= 0:
                ctrl_mean[di, hi] = cm[c]
    mu = ctrl_mean.mean(axis=1, keepdims=True)
    sd = ctrl_mean.std(axis=1, keepdims=True) + 1e-6
    ctrl_feat_all = (ctrl_mean - mu) / sd

    # ---- training items (LOCO excludes the holdout) ----
    tr_di, tr_cond, tr_mean = [], [], []
    holdout_di = None
    for di, cd in enumerate(ctx):
        if args.holdout and cd["name"] == args.holdout:
            holdout_di = di
            continue
        for cond, mean in cd["items"]:
            tr_di.append(di)
            tr_cond.append(cond)
            tr_mean.append(mean)
    print(f"[data] train perts {len(tr_cond)} | holdout: "
          f"{args.holdout or 'none'}", flush=True)
    if not tr_cond:
        raise RuntimeError("training items are empty (did the holdout exclude "
                           "every context?)")
    if args.max_items and args.max_items < len(tr_cond):
        rng = np.random.default_rng(args.seed)
        keep = sorted(rng.choice(len(tr_cond), args.max_items, replace=False))
        tr_di = [tr_di[i] for i in keep]
        tr_cond = [tr_cond[i] for i in keep]
        tr_mean = [tr_mean[i] for i in keep]
        print(f"[data] smoke: truncated to {len(tr_cond)} items", flush=True)

    # ---- residual targets (train-only common core, same as train_p3) ----
    def targets_of(idxs):
        T = np.zeros((len(idxs), len(hvg_rows)), dtype=np.float32)
        rows_p = np.zeros(len(idxs), dtype=np.int64)
        for k, k_item in enumerate(idxs):
            di = tr_di[k_item]
            cond = tr_cond[k_item]
            cp = kd["pert_labels"][di]
            ii = np.where(cp == cond)[0]
            mean = kd["X_pert"][di][ii].mean(axis=0)
            r2c = {int(r): c for c, r in enumerate(kd["row_of_gene"][di])}
            for hi, hr in enumerate(hvg_rows):
                c = r2c.get(hr)
                if c is not None:
                    T[k, hi] = mean[c]
            rows_p[k] = resolve_pert_row(sym2row, cond)
        return T, rows_p

    all_idx = list(range(len(tr_cond)))
    T_tr, rows_tr = targets_of(all_idx)
    fc_tr = T_tr - ctrl_mean[np.array(tr_di)]
    common_fc = fc_tr.mean(axis=0)
    dev_tr = fc_tr - common_fc
    print(f"[P2.3-B] common fc (train-only): std={common_fc.std():.4f} "
          f"max|.|={np.abs(common_fc).max():.4f}", flush=True)

    # ---- v2-1a gating: one scalar per item (the ctrl expression of the
    # target gene in its own context) ----
    pe_list, pe_in_panel = [], 0
    for k, r in enumerate(rows_tr):
        di = tr_di[k]
        col = ctx[di]["sym2col"].get(sym_by_row.get(int(r)))
        if col is not None:
            pe_list.append(float(ctx[di]["cm"][col]))
            pe_in_panel += 1
        else:
            pe_list.append(0.0)
    print(f"[v2-1a] target gene inside the context panel "
          f"{pe_in_panel}/{len(pe_list)} | "
          f"expr median {np.median(pe_list):.3f}", flush=True)

    # ---- model ----
    esm = load_esm_matrix(args.esm_table)
    model = CellEmbDeviationModel(esm, hvg_rows, n_ds=max(2, n_ds)).to(device)
    cell_enc = CellStateEncoder(esm.shape[1]).to(device)
    n_par = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in cell_enc.parameters())
    print(f"[model] params {n_par/1e6:.2f}M | d_cell {D_CELL} | hvg {len(hvg_rows)}",
          flush=True)

    # per-context constant tensors (panel ESM row indices + z)
    ctx_rows_t = [torch.tensor(cd["rows"], dtype=torch.long).to(device) for cd in ctx]
    ctx_z_t = [torch.from_numpy(cd["z"]).float().to(device) for cd in ctx]

    def ctx_embs():
        embs = [cell_enc(model.esm_table, ctx_rows_t[di], ctx_z_t[di]) for di in range(n_ds)]
        return torch.stack(embs)                              # [n_ds, d_cell]

    P_emb = ctx_embs().detach()                               # initial (used as
                                                              # a correlation
                                                              # reference)
    C = np.corrcoef(P_emb.cpu().numpy())
    print("[ctx] cell-embedding correlation matrix (init):", flush=True)
    print(np.round(C, 2), flush=True)

    di_t = torch.tensor(tr_di, dtype=torch.long)
    dev_t = torch.from_numpy(dev_tr).float().to(device)
    pr_t = torch.tensor(rows_tr, dtype=torch.long).to(device)
    cf_t = torch.from_numpy(ctrl_feat_all[np.array(tr_di)]).float().to(device)
    pe_t = torch.tensor(pe_list, dtype=torch.float32).to(device)

    opt = torch.optim.AdamW(list(model.parameters()) + list(cell_enc.parameters()),
                            lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(pr_t)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            embs = ctx_embs()                                 # recomputed every
                                                              # step (the params
                                                              # are updating)
            pred = model(pr_t[idx], di_t[idx], cf_t[idx],
                         pert_ctrl_expr=pe_t[idx],
                         cell_emb=embs[di_t[idx]])
            loss = nn.functional.mse_loss(pred, dev_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(cell_enc.parameters()), 1.0)
            opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                embs = ctx_embs()
                outs = []
                for lo in range(0, n, 256):                   # chunked: avoids OOM
                    hi = min(lo + 256, n)
                    outs.append(model(pr_t[lo:hi], di_t[lo:hi], cf_t[lo:hi],
                                      pert_ctrl_expr=pe_t[lo:hi],
                                      cell_emb=embs[di_t[lo:hi]]).cpu().numpy())
                p = np.concatenate(outs)
                rhos = [np.corrcoef(dev_tr[b], p[b])[0, 1]
                        if dev_tr[b].std() > 1e-6 and p[b].std() > 1e-6 else 0.0
                        for b in range(len(dev_tr))]
            print(f"ep{ep+1}: train_loss {loss.item():.4f} | train pearson_dev "
                  f"{np.mean(rhos):.4f} | {time.time()-t0:.0f}s", flush=True)

    torch.save({"model": model.state_dict(), "cell_enc": cell_enc.state_dict(),
                "hvg_rows": hvg_rows, "common_fc": common_fc,
                "context_names": args.context_names},
               os.path.join(args.out_dir, "ckpt_cell_emb.pt"))

    # ---- LOCO evaluation: zero-shot vs nearest-context fallback (in the
    # production 2000-HVG space) ----
    results = {}
    if holdout_di is not None:
        cd = ctx[holdout_di]
        model.eval()
        with torch.no_grad():
            embs = ctx_embs().cpu().numpy()
        sims = [(args.context_names[j], float(
            np.corrcoef(embs[holdout_di], embs[j])[0, 1])) for j in range(n_ds)
            if j != holdout_di]
        nearest, sim = max(sims, key=lambda x: x[1])
        nj = args.context_names.index(nearest)
        print(f"[loco] holdout {cd['name']} | nearest context: {nearest} "
              f"(emb cos={sim:.2f})", flush=True)

        # ground truth of the holdout items (same protocol)
        cp = kd["pert_labels"][holdout_di]
        h_items = [(cond, np.where(cp == cond)[0]) for cond, _ in cd["items"]]
        r2c = {int(r): c for c, r in enumerate(kd["row_of_gene"][holdout_di])}
        cm_h = kd["X_ctrl"][holdout_di].mean(axis=0)
        cf_h = ctrl_feat_all[holdout_di]

        def eval_ctx(mode):
            # assemble all holdout items in a batch (looping item by item is
            # too slow and prone to hanging)
            prs, pes, tvs = [], [], []
            for cond, ii in h_items:
                row = resolve_pert_row(sym2row, cond)
                prs.append(row)
                mean = kd["X_pert"][holdout_di][ii].mean(axis=0)
                tv_raw = np.zeros(len(hvg_rows), dtype=np.float32)
                for hi, hr in enumerate(hvg_rows):
                    c = r2c.get(hr)
                    if c is not None:
                        tv_raw[hi] = mean[c]
                tvs.append(tv_raw - common_fc)
                col = cd["sym2col"].get(sym_by_row.get(int(row)))
                pes.append(float(cm_h[col]) if col is not None else 0.0)
            pr_t_h = torch.tensor(prs, dtype=torch.long).to(device)
            pe_t_h = torch.tensor(pes, dtype=torch.float32).to(device)
            cf_v = torch.from_numpy(cf_h[None, :]).float().to(device)
            outs = []
            di_v = (torch.zeros(len(prs), dtype=torch.long) if mode == "zero_shot"
                    else torch.full((len(prs),), nj, dtype=torch.long))
            with torch.no_grad():
                if mode == "zero_shot":
                    emb = cell_enc(model.esm_table, ctx_rows_t[holdout_di],
                                   ctx_z_t[holdout_di]).unsqueeze(0)  # [1, d_cell]
                for lo in range(0, len(prs), 256):
                    hi = min(lo + 256, len(prs))
                    kw = {"pert_ctrl_expr": pe_t_h[lo:hi]}
                    if mode == "zero_shot":
                        kw["cell_emb"] = emb.expand(hi - lo, -1)   # align to the chunk batch
                    outs.append(model(pr_t_h[lo:hi], di_v[lo:hi], cf_v.expand(hi - lo, -1),
                                      **kw).cpu().numpy())
            P_pred = np.concatenate(outs)
            T_true = np.stack(tvs)
            rhos = [np.corrcoef(T_true[b], P_pred[b])[0, 1]
                    if T_true[b].std() > 1e-6 and P_pred[b].std() > 1e-6 else np.nan
                    for b in range(len(T_true))]
            rhos = [r for r in rhos if np.isfinite(r)]
            return float(np.mean(rhos)), len(rhos)

        zs, n_zs = eval_ctx("zero_shot")
        fb, _ = eval_ctx("fallback")
        results = dict(holdout=cd["name"], nearest=nearest, nearest_sim=sim,
                       zero_shot_pearson_dev=zs, fallback_pearson_dev=fb,
                       delta=zs - fb, n_eval=n_zs, n_hvg=len(hvg_rows),
                       gate="PASS" if (zs - fb >= 0.02 and zs > 0.15) else "FAIL",
                       gate_rule="zero-shot - fallback >= +0.02 and zero-shot > 0.15; "
                                 "FAIL = permanently closed (Route A has already "
                                 "removed the data-conditions excuse)")
        print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)
    json.dump(results, open(os.path.join(args.out_dir, "loco_results.json"), "w"),
              indent=2, ensure_ascii=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
