"""VCPE-rna P2.3-B training: direct per-gene deviation head (no SE backbone).

Protocol (inherited from P2.1/P2.2, kept intact):
  true_fc      = pert_mean_hvg - dataset_ctrl_mean_hvg
  common_fc    = mean over TRAIN perturbations only (no leakage)
  true_dev     = true_fc - common_fc   <- the model predicts THIS directly
  pred_fc      = pred_dev + common_fc  (for full-fc metrics)

Three item sets, and they do not overlap:
  gradient     = training items MINUS the inner-validation slice
  inner-val    = selects the epoch; never reported as performance
  test         = scored exactly once, at the selected epoch
The control baselines are fitted on the gradient set too, so every row of the
results table has seen the same items. Before this was enforced, the inner
validation slice was also trained on -- so the "selection" curve was a training
curve and the head had ~18% more fitting data than the external baselines it was
compared with (docs/ERRATA.md E16). `--train_on_inner_val` restores the old
behaviour for measurement.

Eval reports full-fc metrics (mse_DE / pearson_delta / top50) AND dev metrics
(mse_dev / pearson_dev / top50_dev) PLUS a built-in conditioning ablation:
each epoch, pred_dev is recomputed with the target ESM2 vector zeroed and
shuffled; ablation_r = corr(real, zero). P2/P2.1/P2.2 all showed 1.000 (dead).
If conditioning is alive, ablation_r drops well below 1.

No SE / flash-attention / MAP repo / env vars required. Runs in minutes.
"""
import argparse
import hashlib
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
from baselines import run_all, select_control_hyperparams  # noqa: E402
from eval_metrics import (  # noqa: E402
    direction_agreement, enrichment_factor, paired_bootstrap_per_item,
    per_item_correlation, per_item_correlation_detail, per_item_spearman,
    pooled_correlation, top_k_overlap)


RESUME_FILE = "resume_state.pt"

# The protocol fields a resumed run must agree with the interrupted one about.
# Resuming into a different experiment would splice two runs into one number,
# which is the traceability failure of ERRATA E11 with a longer fuse.
RESUME_FINGERPRINT_ARGS = (
    "data_dirs", "esm_table", "string_neighbors", "rna_encoder_ckpt", "n_hvg",
    "hvg_from", "split_by", "test_frac", "min_cells", "seed", "inner_val_frac",
    "train_on_inner_val", "use_mask", "batch_size", "lr", "weight_decay",
    "d_model", "epochs")


def resume_fingerprint(args, data, fit_pos):
    """What must match for a resumed run to be the same run.

    Includes the derived split, not only the arguments that produced it: an
    ingest script or a data file could change underneath an interrupted run and
    rebuild a different split from identical flags.
    """
    def _h(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    fp = {k: getattr(args, k, None) for k in RESUME_FINGERPRINT_ARGS}
    fp["hvg_rows"] = _h(np.asarray(data["hvg_rows"], dtype=np.int64))
    fp["rows_tr"] = _h(np.asarray(data["rows_tr"], dtype=np.int64))
    fp["rows_te"] = _h(np.asarray(data["rows_te"], dtype=np.int64))
    fp["is_inner_val"] = _h(np.asarray(data["is_inner_val"], dtype=np.int8))
    fp["fit_pos"] = _h(np.asarray(fit_pos, dtype=np.int64))
    return fp


def save_resume_state(path, *, epochs_done, args, model, opt, sched, best_pd,
                      best_epoch, fingerprint):
    """Write the state atomically: a torn state file is worse than none.

    The RNG states go in too, so a resumed run continues the same stream rather
    than starting a statistically different one at the same epoch number.
    """
    state = dict(
        epochs_done=int(epochs_done), epochs_budget=int(args.epochs),
        best_pd=float(best_pd), best_epoch=int(best_epoch),
        fingerprint=fingerprint,
        model_state_dict=model.state_dict(), optimizer=opt.state_dict(),
        scheduler=sched.state_dict(),
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        provenance=stamp(args=args))
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_resume_state(path, args, fingerprint):
    """Load and verify. A mismatch refuses rather than resuming into another run.

    The budget is part of the fingerprint on purpose. Resuming after an
    infrastructure failure is not a budget change -- and `--epochs 150` resumed
    as `--epochs 200` is how a declared budget quietly becomes "extend until the
    favoured model wins", which RESULTS_REAL.md names as the abuse to avoid.
    """
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st.get("epochs_budget") != args.epochs:
        raise SystemExit(
            f"this run was declared at {st.get('epochs_budget')} epochs and you "
            f"passed --epochs {args.epochs}. Resuming is not a budget change: "
            f"re-declare the budget in a new --out_dir, so the longer run is a "
            f"new run with its own record.")
    old, new = st.get("fingerprint", {}), fingerprint
    diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
            if old.get(k) != new.get(k)}
    if diff:
        raise SystemExit(
            "the state in this --out_dir belongs to a different experiment, so "
            "resuming it would splice two runs into one number. Differences "
            f"(stored, now): {diff}")
    return st


def _control_hyperparameters(name, args, selected):
    """What each control actually ran at, in its own row of the report.

    A control quoted without its hyperparameter is as uncomparable as a score
    quoted without its estimator: `ridge_esm2` at alpha=1.0 and `ridge_esm2_sel`
    at a selected alpha are different controls, and the difference between them
    is part of what the table shows.
    """
    sel = selected or {}
    if name == "ridge_esm2":
        return {"alpha": float(args.ridge_alpha), "chosen_by": "fixed default"}
    if name == "knn_esm2":
        return {"k": int(args.knn_k), "chosen_by": "fixed default"}
    if name == "ridge_esm2_sel":
        return {"alpha": sel.get("ridge_alpha"), "chosen_by": "inner_val"}
    if name == "knn_esm2_sel":
        return {"k": sel.get("knn_k"), "chosen_by": "inner_val"}
    if name == "rff_ridge_esm2_sel":
        return {"alpha": sel.get("rff_alpha"),
                "gamma_mult": sel.get("rff_gamma_mult"),
                "n_features": sel.get("rff_n_features"),
                "seed": sel.get("rff_seed"), "chosen_by": "inner_val"}
    return {"chosen_by": "no hyperparameters"}


def secondary_metrics(true, pred, mask):
    """Metrics a Pearson correlation does not imply, under their own names.

    `pearson_dev` is the headline and stays the headline. But a model can
    correlate well and still get the sign wrong on the genes with the largest
    responses, and ERRATA E2 was a top-k enrichment whose chance level was
    misscaled, so both are worth carrying beside it -- computed once here, for
    the model and for every control, from the shared definitions.
    """
    return {
        "spearman_dev": per_item_spearman(true, pred, mask),
        "enrichment_top5pct_dev": enrichment_factor(true, pred, k_frac=0.05,
                                                    mask=mask),
        "direction_agreement_top10pct_dev": direction_agreement(
            true, pred, mask=mask, top_frac=0.10),
        "secondary_metric_chance_levels": {
            "spearman_dev": 0.0, "enrichment_top5pct_dev": 1.0,
            "direction_agreement_top10pct_dev": 0.5},
    }
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import stamp, write_json  # noqa: E402


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
    p.add_argument("--train_on_inner_val", action="store_true", default=False,
                   help="reproduce the pre-fix behaviour in which the inner "
                        "validation items were ALSO used as training batches, so "
                        "the epoch was selected on data the model had fitted "
                        "(docs/ERRATA.md E16). With this flag the selection curve "
                        "is a training curve: it rises for as long as the budget "
                        "lasts, which is why every long run here selected its "
                        "final epoch. Kept so the effect can be measured by "
                        "flipping one flag; it is not a configuration to train "
                        "for release.")
    p.add_argument("--no_mask", dest="use_mask", action="store_false", default=True,
                   help="reproduce the pre-v4 behaviour in which HVG columns a "
                        "dataset never measured entered the loss and every metric "
                        "unmasked. Those columns carry a per-dataset CONSTANT "
                        "target (dev == -common_fc) with no perturbation-specific "
                        "content, so scoring them rewards reproducing that "
                        "constant. Provided so the size of that inflation can be "
                        "measured on real data by flipping this one flag "
                        "(docs/ERRATA.md E6); it is not a configuration to train "
                        "for release.\n")
    p.add_argument("--knn_k", type=int, default=10,
                   help="neighbours for the ESM2 retrieval control baseline, as a "
                        "fixed default. The `knn_esm2_sel` row selects k on the "
                        "inner-validation slice instead; this one is kept so the "
                        "name means in a new table what it meant in an old one.")
    p.add_argument("--ridge_alpha", type=float, default=1.0,
                   help="ridge penalty for the fixed-default `ridge_esm2` row. "
                        "The `ridge_esm2_sel` row selects it on inner validation.")
    p.add_argument("--no_control_selection", action="store_true", default=False,
                   help="skip inner-validation selection for the control "
                        "baselines, leaving only the fixed-default rows. Selection "
                        "costs a handful of closed-form solves; this exists for "
                        "reproducing a table made before it.")
    p.add_argument("--rff_features", type=int, default=1024,
                   help="random Fourier features for the non-linear control. This "
                        "is the rung between a linear map and the head: a generic "
                        "non-linear map on identical features, fitted in closed "
                        "form. If it matches the head, the head's margin over "
                        "ridge is non-linearity rather than architecture.")
    p.add_argument("--n_boot_items", type=int, default=2000,
                   help="bootstrap replicates for the paired item-level interval "
                        "against each control. 0 disables it.")
    p.add_argument("--resume", action="store_true", default=False,
                   help="write a resume state after every epoch and continue from "
                        "it if one is already in --out_dir. The pre-declared "
                        "150-epoch run in RESULTS_REAL.md was never completed "
                        "because this container restarted twice mid-run, and "
                        "without this a restart costs the whole run. The stored "
                        "budget, the split and the data fingerprint must match, "
                        "so a resume cannot silently become a different "
                        "experiment or a longer budget.")
    p.add_argument("--stop_after_epochs", type=int, default=0,
                   help="run at most this many epochs in THIS invocation, then "
                        "save the resume state and exit without scoring the "
                        "held-out set. Requires --resume. It is how a declared "
                        "budget is completed on a host that cannot hold the whole "
                        "run: the same command, run again, continues where it "
                        "stopped. An incomplete run writes no final report, so a "
                        "partial budget cannot be quoted as a result.")
    p.add_argument("--hvg_from", choices=("train", "all"), default="train",
                   help="which perturbations the response panel is ranked on. train "
                        "(default) is correct. all reproduces the earlier behaviour, "
                        "in which the 2000 genes the model is scored on were chosen "
                        "by variance across every perturbation including the held-out "
                        "ones -- feature selection with the test split visible "
                        "(docs/ERRATA.md E15). Kept so that effect can be measured.")
    p.add_argument("--d_model", type=int, default=256)
    return p.parse_args()


# ---------------- data ----------------

def split_items(args, kd, sym2row):
    """All usable (dataset, condition) items and their train/test split.

    Separated out so that it can run before the response panel is chosen; see
    build_dev_data.
    """
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
    return train_items, test_items


def build_dev_data(args):
    sym2row, esm_dim = build_symbol2row(args.esm_table)
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)

    # The split is decided BEFORE the response panel, so that the panel can be
    # ranked on training perturbations only (docs/ERRATA.md E15). It needs nothing
    # but the perturbation labels, and it draws from a generator of its own, so
    # moving it here leaves the split itself unchanged.
    train_items, test_items = split_items(args, kd, sym2row)
    if getattr(args, "hvg_from", "train") == "train":
        tr_set = set(train_items)
        keep_rows = []
        for di in range(len(kd["X_pert"])):
            cp = kd["pert_labels"][di]
            keep_rows.append(np.array([k for k, c in enumerate(cp)
                                       if (di, str(c)) in tr_set], dtype=np.int64))
        hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg, keep_rows=keep_rows)
        print(f"[hvg] panel ranked on {sum(len(k) for k in keep_rows)} TRAINING "
              f"perturbations only", flush=True)
    else:
        hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg)
        print("[hvg] WARNING: panel ranked on ALL perturbations including the held-out "
              "genes, so the genes the model is scored on were chosen with the test "
              "split visible. Kept only to measure that effect (ERRATA E15); use "
              "--hvg_from train.", flush=True)

    # dataset-level control mean over HVG (deterministic ctrl baseline)
    ctrl_mean = np.zeros((len(kd["X_ctrl"]), len(hvg_rows)), dtype=np.float32)
    for di, (Xc, rows) in enumerate(zip(kd["X_ctrl"], kd["row_of_gene"])):
        cm = Xc.mean(axis=0)
        row2col = {int(r): c for c, r in enumerate(rows)}
        cols = np.array([row2col.get(int(hr), -1) for hr in hvg_rows], dtype=np.int64)
        ok = cols >= 0
        ctrl_mean[di, ok] = cm[cols[ok]]
    # z-score across genes within each dataset (feature scale for the MLP)
    mu, sd = ctrl_mean.mean(axis=1, keepdims=True), ctrl_mean.std(axis=1, keepdims=True) + 1e-6
    ctrl_feat_all = (ctrl_mean - mu) / sd


    # Per-dataset lookups, built once. Both of these used to be rebuilt inside the
    # per-item loops below, which is quadratic in a way that only shows up on a real
    # dataset: 2204 perturbations x an 8749-entry dict, and 2204 recomputations of a
    # 3000 x 8749 control mean. On Replogle RPE1 that is tens of billions of
    # redundant float operations before a single training step.
    row2col_ds = [{int(r): c for c, r in enumerate(rows)}
                  for rows in kd["row_of_gene"]]
    ctrl_mean_full = [Xc.mean(axis=0) for Xc in kd["X_ctrl"]]
    hvg_col_ds = [np.array([r2c.get(int(hr), -1) for hr in hvg_rows], dtype=np.int64)
                  for r2c in row2col_ds]

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
            cols = hvg_col_ds[di]                      # [n_hvg], -1 where unmeasured
            ok = cols >= 0
            T[k, ok] = mean[cols[ok]]
            M[k] = ok
            rows_p[k] = resolve_pert_row(sym2row, cond)
        return T, rows_p, M

    T_tr, rows_tr, mask_tr = targets_of(train_items)
    T_te, rows_te, mask_te = targets_of(test_items)
    print(f"[mask] measured HVG fraction: train {mask_tr.mean():.3f} "
          f"test {mask_te.mean():.3f} (unmeasured columns are excluded from "
          f"loss and metrics)", flush=True)

    # V2-1a: ctrl expression of the target gene itself (raw log1p, feeds the
    # self-response gate).
    # Target gene inside the panel of its own dataset -> take that column's ctrl
    # mean; outside the panel -> 0.0
    # (in that case the target is usually not in the HVG response panel either,
    # so there is no is_tgt row, the gate does not act, and this is safe).
    def pert_expr_of(split_items):
        E = np.zeros(len(split_items), dtype=np.float32)
        for k, (di, cond) in enumerate(split_items):
            r = resolve_pert_row(sym2row, cond)
            if r < 0:
                continue
            c = row2col_ds[di].get(int(r))
            if c is not None:
                E[k] = float(ctrl_mean_full[di][c])
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
def evaluate(model, data, split, device, esm_override_mode=None, use_mask=True,
             return_pred=False):
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
    # --no_mask restores the pre-v4 behaviour so the inflation can be measured.
    # use_mask=False restores the pre-v4 behaviour so the inflation from
    # unmeasured columns can be measured (docs/ERRATA.md E6).
    mask = mask_all if use_mask else np.ones_like(mask_all, dtype=bool)

    pred_fc = pred_dev + data["common_fc"]
    out = {}
    out["measured_fraction"] = float(mask.mean())

    def _masked_mse(pred, true):
        d2 = ((pred - true) ** 2) * mask
        denom = np.maximum(mask.sum(axis=1), 1)
        return float((d2.sum(axis=1) / denom).mean())

    def _per_item(pred, true, k=50):
        """Per-perturbation Pearson and top-k overlap over measured columns only.

        Delegates to `eval_metrics`, which the baselines are also scored by. This
        used to be a second implementation living here, differing from the shared
        one only in its degeneracy threshold (1e-6 against 1e-12) -- so the model
        and the controls in the same table were scored by two implementations of
        `pearson_dev`. That is E4's shape one level down, and
        `tests/test_metrics.py` now pins the two as equal on profiles whose std
        is not inside that window, which is every real one.

        The top-k term is skipped for items with fewer than 2k measured columns:
        taking the top n of n gives 1.0 by construction, which reads as a perfect
        score while carrying no information.
        """
        d = per_item_correlation_detail(true, pred, mask)
        return (d["value"], top_k_overlap(true, pred, k=k, mask=mask, min_ratio=2),
                d["n_scored"])

    out["mse_DE"] = _masked_mse(pred_fc, fc)
    out["mse_DE_baseline_ctrl"] = _masked_mse(np.zeros_like(fc), fc)
    out["pearson_delta"], out["top50_deg_overlap"], _ = _per_item(pred_fc, fc)
    # dev metrics
    out["mse_dev"] = _masked_mse(pred_dev, dev)
    pd_, td_, n_scored = _per_item(pred_dev, dev)
    out["pearson_dev"], out["top50_dev"] = pd_, td_
    out["n_scored_perturbations"] = n_scored
    # Metrics that a correlation does not imply, computed for the model here and
    # for every control in the final table, so the rows are comparable.
    out.update(secondary_metrics(dev, pred_dev, mask))
    _d = per_item_correlation_detail(dev, pred_dev, mask)
    out["n_degenerate_perturbations"] = _d["n_degenerate"]
    out["n_skipped_too_few_features"] = _d["n_skipped_too_few_features"]
    # Estimator label, so that this number is never silently compared against
    # eval_fair.py's flattened pooled correlation -- they are different
    # estimators and differ by ~1.5x on the same checkpoint.
    out["pearson_dev_estimator"] = "mean_over_perturbations_of_within_perturbation_r"
    if return_pred:
        return out, pred_dev
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
    mask_tr_t = torch.from_numpy(
        data["mask_tr"] if args.use_mask
        else np.ones_like(data["mask_tr"])).float().to(device)
    ds_tr = torch.tensor([di for di, _ in data["train_items"]], dtype=torch.long)
    rows_tr = torch.tensor(data["rows_tr"], dtype=torch.long)
    cf_tr = torch.from_numpy(data["ctrl_feat_all"][ds_tr.numpy()]).float().to(device)
    pe_tr = torch.from_numpy(data["pert_expr_tr"]).float().to(device)
    rna_tr = (data["rna_embs"][:len(data["train_items"])].to(device)
              if model.proj_r is not None else None)

    # ---- which training items the gradient actually sees ----
    # The inner-validation slice exists to select the epoch. If the model also
    # trains on it, that slice is training data and its correlation rises with
    # the training loss, so "selection" picks whatever epoch the budget ends on
    # -- which is what every long run here did (epochs 78/80, 150/150). It also
    # silently gave the head ~18% more training items than GEARS and CPA, which
    # `baseline_common.inner_split_items` hands the inner-TRAIN items only while
    # documenting the footing as identical. See docs/ERRATA.md E16.
    is_val = np.asarray(data["is_inner_val"], dtype=bool)
    fit_pos = (np.arange(len(data["train_items"])) if args.train_on_inner_val
               else np.flatnonzero(~is_val))
    if not args.train_on_inner_val:
        # Mechanical, not a promise in a comment: the gradient set and the
        # selection set must not intersect.
        assert not is_val[fit_pos].any(), \
            "inner-validation items reached the training batches"
    if len(fit_pos) == 0:
        raise SystemExit("every training item landed in the inner-validation "
                         "split, so there is nothing to fit on; lower "
                         "--inner_val_frac")
    fit_pos_t = torch.as_tensor(fit_pos, dtype=torch.long)
    print(f"[split] gradient items: {len(fit_pos)} of {len(data['train_items'])} "
          f"training items"
          + ("  (--train_on_inner_val: the selection slice is ALSO trained on, "
             "so the selection curve is a training curve -- ERRATA E16)"
             if args.train_on_inner_val else
             "  (inner-val held out of the gradient, as the selection requires)"),
          flush=True)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    # The log is append-only, so a re-run into an existing out_dir concatenates two
    # runs whose epoch numbers both start at 1 -- which is what happened when a
    # killed 80-epoch run was resumed and left a log reading 1..74 then 1..80,
    # unsplittable by anything reading it. ERRATA cites train_log.jsonl as
    # evidence, so an ambiguous boundary is an E11-class traceability problem
    # rather than cosmetic. History is kept; the boundary is made explicit and
    # machine-readable instead. Consumers should take the records after the LAST
    # run_start.
    n_prior = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            n_prior = sum(1 for _ in f)
    with open(log_path, "a") as f:
        f.write(json.dumps(dict(
            run_start=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            prior_records_in_this_file=n_prior,
            epochs_planned=args.epochs,
            note="records after this line belong to one run; epoch numbering "
                 "restarts here",
            provenance=stamp(args=args))) + "\n")
    if n_prior:
        print(f"[log] {log_path} already held {n_prior} records from an earlier "
              f"run; a run_start boundary was written so the two cannot be "
              f"confused", flush=True)
    best_pd, best_epoch = -float('inf'), -1
    ckpt_path = os.path.join(args.out_dir, 'ckpt_p3_best_dev.pt')
    n = len(fit_pos_t)

    # ---- resume ----
    resume_path = os.path.join(args.out_dir, RESUME_FILE)
    fingerprint = resume_fingerprint(args, data, fit_pos)
    start_epoch = 0
    if args.stop_after_epochs and not args.resume:
        raise SystemExit("--stop_after_epochs needs --resume, or the epochs run "
                         "in this invocation would be lost rather than continued")
    if args.resume and os.path.exists(resume_path):
        st = load_resume_state(resume_path, args, fingerprint)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        torch.set_rng_state(st["torch_rng"])
        np.random.set_state(st["numpy_rng"])
        start_epoch = int(st["epochs_done"])
        best_pd, best_epoch = float(st["best_pd"]), int(st["best_epoch"])
        print(f"[resume] continuing at epoch {start_epoch + 1}/{args.epochs}; "
              f"best inner-val so far {best_pd:.4f} at ep{best_epoch}. The RNG "
              f"states were restored, so this continues the interrupted run "
              f"rather than starting a statistically different one.", flush=True)
    elif args.resume:
        print(f"[resume] no state in {args.out_dir}; starting fresh and saving "
              f"state after every epoch", flush=True)

    stop_at = (min(args.epochs, start_epoch + args.stop_after_epochs)
               if args.stop_after_epochs else args.epochs)
    for epoch in range(start_epoch, stop_at):
        model.train()
        t0, run = time.time(), []
        perm = fit_pos_t[torch.randperm(n)]
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
            ev_in = evaluate(model, data, "inner_val", device,
                             use_mask=args.use_mask)
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
        if args.resume:
            save_resume_state(resume_path, epochs_done=epoch + 1, args=args,
                              model=model, opt=opt, sched=sched, best_pd=best_pd,
                              best_epoch=best_epoch, fingerprint=fingerprint)

    if stop_at < args.epochs:
        # No final report: a partial budget must not be quotable as a result.
        print(f"\nSTOPPED at {stop_at}/{args.epochs} epochs as asked. The "
              f"held-out set was NOT scored and no final_report.json was "
              f"written, because a number from a partial budget is not a result. "
              f"Run the same command again to continue from epoch {stop_at + 1}.",
              flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(dict(
                stopped_early=True, epochs_done=stop_at,
                epochs_budget=args.epochs,
                note="incomplete run; no held-out scoring")) + "\n")
        return

    # ---- single, final scoring of the held-out split, at the selected epoch ----
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    load_dev_state(model, ck)
    model.to(device).eval()
    final, model_pred_te = evaluate(model, data, "test", device,
                                    use_mask=args.use_mask, return_pred=True)
    final["selected_epoch"] = best_epoch
    final["inner_val_pearson_dev_at_selection"] = float(best_pd)
    final["split_by"] = args.split_by
    # Protocol fields: which items were fitted, and by what rule. A number quoted
    # without them is not comparable to another number (ERRATA E11, E16), and the
    # controls below are fitted on the same set, so one field covers every row of
    # the table.
    final["trained_on_inner_val"] = bool(args.train_on_inner_val)
    final["n_train_items"] = int(len(data["train_items"]))
    final["n_fit_items"] = int(len(fit_pos))
    final["n_inner_val_items"] = int(is_val.sum())
    final["n_test_items"] = int(len(data["test_items"]))
    # Whether the budget, not the data, ended the run. Selecting the last epoch
    # means the model was still improving when the budget expired, so the number
    # is a floor under that budget rather than a converged estimate -- stated
    # mechanically here because RESULTS_REAL.md had to say it in prose three
    # times (epochs 78/80, 150/150).
    final["selection_hit_budget"] = bool(best_epoch >= args.epochs)
    final["epochs_budget"] = int(args.epochs)
    if final["selection_hit_budget"]:
        print(f"\nNOTE: the selected epoch IS the budget ({best_epoch}/"
              f"{args.epochs}), so this model had not converged. Read the held-out "
              f"number as a floor under this budget, not as a converged estimate.",
              flush=True)

    # ---- control baselines on the same split, same mask, same metrics ----
    # Without these there is no evidence that the learned head does more than
    # retrieval over the frozen ESM2 embeddings it is conditioned on.
    esm_np = model.esm_table.detach().cpu().numpy()
    # The graph-only control needs the same is_neighbor indicator the model sees,
    # otherwise a gain from the model's network channel has nothing to be measured
    # against and would look like an architectural win rather than extra input.
    is_nb_tr = is_nb_te = None
    if getattr(model, "neighbor_table", None) is not None:
        with torch.no_grad():
            # indicator_features returns (is_nb, is_tgt) -- neighbour FIRST.
            # Indexing [1] here would hand the control the TARGET indicator and
            # silently measure the wrong thing, so the order is asserted below.
            is_nb_tr, is_tgt_tr = model.indicator_features(
                torch.as_tensor(data["rows_tr"], dtype=torch.long))
            is_nb_te, _ = model.indicator_features(
                torch.as_tensor(data["rows_te"], dtype=torch.long))
            is_nb_tr = is_nb_tr.cpu().numpy()
            is_nb_te = is_nb_te.cpu().numpy()
            # is_tgt marks the perturbed gene itself, so it can fire at most once
            # per row; is_nb is a neighbourhood and generally fires more. If this
            # trips, the tuple order flipped.
            assert is_tgt_tr.cpu().numpy().sum(axis=1).max() <= 1, \
                "indicator_features order looks flipped: the second element is " \
                "firing on many genes per item, which is is_nb, not is_tgt"
        print(f"[baselines] graph-only control enabled: is_neighbor covers "
              f"{is_nb_te.mean() * 100:.2f}% of held-out (item, gene) cells",
              flush=True)
    # Without a graph, neighbor_prior is *identical* to train_mean by construction.
    # Reporting both would put a duplicate row in the table that reads like an
    # independent control corroborating the first, so it is requested only when a
    # graph is actually present.
    which = ["zero", "train_mean", "knn_esm2", "ridge_esm2"]
    if is_nb_te is not None:
        which.append("neighbor_prior")

    # ---- the controls get the selection discipline the head gets ----
    # alpha=1.0 and k=10 were fixed defaults for every number this repository has
    # published, while the head's epoch is selected on the inner split and its lr
    # / width / decay were tuned over many runs on this data. A control denied its
    # one knob is a weaker control than it looks. Chosen here on the
    # inner-validation slice only -- the held-out set is not consulted, and the
    # full grid goes into the report so the choice is auditable.
    val_pos = np.flatnonzero(is_val)
    selected = None
    if len(val_pos) >= 2 and not args.no_control_selection:
        print(f"\n[controls] selecting hyperparameters on {len(val_pos)} "
              f"inner-validation items (held-out set not consulted)", flush=True)
        selected = select_control_hyperparams(
            data["dev_tr"][fit_pos], esm_np[data["rows_tr"][fit_pos]],
            data["dev_tr"][val_pos], esm_np[data["rows_tr"][val_pos]],
            mask_fit=(data["mask_tr"][fit_pos] if args.use_mask else None),
            mask_val=(data["mask_tr"][val_pos] if args.use_mask else None),
            rff_n_features=args.rff_features, seed=args.seed)
        final["control_selection"] = selected
        which += ["ridge_esm2_sel", "knn_esm2_sel", "rff_ridge_esm2_sel"]
        print(f"[controls] ridge alpha={selected.get('ridge_alpha')} "
              f"knn k={selected.get('knn_k')} "
              f"rff alpha={selected.get('rff_alpha')} "
              f"gamma_mult={selected.get('rff_gamma_mult')}", flush=True)
        edge = [k for k in ("ridge_alpha", "knn_k")
                if selected.get(k + "_at_grid_edge")]
        if edge:
            print(f"[controls] WARNING: {edge} selected at the edge of the grid, "
                  f"so the grid may be what chose them, not the data", flush=True)
    # Fit the controls on exactly the items the head's gradient saw. Handing them
    # the inner-validation items as well would give them ~18% more fitting data
    # than the model they are controls for, which is the same footing defect as
    # E16 with the sign reversed.
    base_preds = run_all(
        data["dev_tr"][fit_pos], esm_np[data["rows_tr"][fit_pos]],
        esm_np[data["rows_te"]],
        mask_tr=(data["mask_tr"][fit_pos] if args.use_mask else None),
        knn_k=args.knn_k, which=which, selected=selected,
        is_nb_tr=(None if is_nb_tr is None else is_nb_tr[fit_pos]),
        is_nb_te=is_nb_te)
    mask_te = data["mask_te"] if args.use_mask else np.ones_like(data["mask_te"])
    dev_te = data["dev_te"]
    final["baselines"] = {}
    for name, pred in base_preds.items():
        row = dict(
            pearson_dev=per_item_correlation(dev_te, pred, mask_te),
            pearson_dev_pooled=pooled_correlation(dev_te, pred, mask_te),
            top50_dev=top_k_overlap(dev_te, pred, k=50, mask=mask_te),
            fit_items=int(len(fit_pos)))
        row.update(secondary_metrics(dev_te, pred, mask_te))
        row["hyperparameters"] = _control_hyperparameters(name, args, selected)
        # The paired item-level interval. It answers a different question from
        # the multi-seed interval in run_seeds.py and does not replace it: this
        # one varies the sample of perturbations within one split, that one
        # varies which genes are held out. On this data ridge's own score moved
        # 0.039 between splits, so the second is the harder question and a
        # narrow interval here is not a settled comparison.
        if args.n_boot_items > 0:
            row["paired_vs_model_items"] = paired_bootstrap_per_item(
                dev_te, model_pred_te, pred, mask=mask_te,
                n_boot=args.n_boot_items, seed=args.seed)
        final["baselines"][name] = row
    final["footing"] = (
        f"model and every control fitted on the same {len(fit_pos)} training "
        f"items; the {int(is_val.sum())} inner-validation items selected the "
        f"epoch and were "
        + ("ALSO fitted (ERRATA E16 behaviour, --train_on_inner_val)"
           if args.train_on_inner_val else "fitted by nobody")
        + f"; {len(data['test_items'])} test items scored once")

    print("\n=== held-out results (scored once, at the selected epoch) ===",
          flush=True)
    print(f"every row fitted on the same {len(fit_pos)} items; "
          f"{len(data['test_items'])} test items; estimator "
          f"{final['pearson_dev_estimator']}", flush=True)
    print(f"{'model':<19} {'pearson':>8} {'spearman':>9} {'EF5%':>6} "
          f"{'dir':>6} {'paired vs model (items)':>30}", flush=True)
    print(f"{'P3 head':<19} {final['pearson_dev']:>8.4f} "
          f"{final['spearman_dev']:>9.4f} "
          f"{final['enrichment_top5pct_dev']:>6.2f} "
          f"{final['direction_agreement_top10pct_dev']:>6.3f} "
          f"{'--':>30}", flush=True)
    for name, m in final["baselines"].items():
        pb = m.get("paired_vs_model_items", {})
        ci = pb.get("ci95") or [float("nan"), float("nan")]
        txt = (f"{pb.get('difference', float('nan')):+.4f} "
               f"[{ci[0]:+.4f},{ci[1]:+.4f}]")
        print(f"{name:<19} {m['pearson_dev']:>8.4f} {m['spearman_dev']:>9.4f} "
              f"{m['enrichment_top5pct_dev']:>6.2f} "
              f"{m['direction_agreement_top10pct_dev']:>6.3f} {txt:>30}",
          flush=True)
    print("chance: spearman 0.0, EF5% 1.0, dir 0.5. The paired interval varies "
          "the SAMPLE OF PERTURBATIONS within this one split; it does not vary "
          "which genes are held out -- run_seeds.py answers that, and it is the "
          "harder question.", flush=True)
    beaten = [n for n, m in final["baselines"].items()
              if final["pearson_dev"] <= m["pearson_dev"]]
    if beaten:
        print(f"\nNOTE: the trained head does NOT beat {beaten} on pearson_dev. "
              f"A conditioned model that loses to retrieval or to a linear map "
              f"over the same embeddings has not been shown to learn "
              f"perturbation-specific biology.", flush=True)
    if selected and "rff_ridge_esm2_sel" in final["baselines"]:
        rff = final["baselines"]["rff_ridge_esm2_sel"]["pearson_dev"]
        lin = final["baselines"].get("ridge_esm2_sel",
                                     final["baselines"]["ridge_esm2"])["pearson_dev"]
        print(f"\n[non-linearity control] a generic non-linear map on the same "
              f"features scores {rff:.4f} against the linear map's {lin:.4f} and "
              f"the head's {final['pearson_dev']:.4f}. Read the head's margin "
              f"over the LINEAR map as architecture only if it also exceeds this "
              f"row; otherwise what the head has is non-linearity.", flush=True)

    ck["metrics"] = final
    torch.save(ck, ckpt_path)
    write_json(os.path.join(args.out_dir, "final_report.json"), final, args=args)
    with open(log_path, "a") as f:
        f.write(json.dumps({"final": final}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
