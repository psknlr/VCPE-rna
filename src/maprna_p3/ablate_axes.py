#!/usr/bin/env python
"""Per-axis conditioning ablation for the P3 deviation head.

Why this script exists
----------------------
Perturbation identity reaches this model through FOUR independent channels:

    1. the target gene's frozen ESM2 vector          (`pert_rows` -> proj_p)
    2. the target transcript's RNA-encoder embedding (`rna_emb` -> proj_r)
    3. `is_target`   -- is this HVG gene the perturbed gene?
    4. `is_neighbor` -- is this HVG gene a STRING neighbour of the perturbed gene?

plus a fifth channel that is not perturbation identity but is still a source of
apparent skill:

    5. `ds_emb` -- which dataset the profile came from, i.e. dataset-level
       response commonality

The single number previously reported as `ablation_r` zeroed channel 1 only. The
P2.3-B report said so explicitly ("the ablation zeroes the ESM2 vector; the
neighbor/self indicator features still carry perturbation identity -- r = 0.28
is the joint effect of three axes; per-axis ablation is future work") and flagged
the ds channel separately ("dataset-level residual commonality may still be
partially exploited via the ds one-hot -- a ds-shuffle ablation would isolate
this if needed"). Neither was ever run, and no hook existed to run them.

This script runs each axis on its own, plus the all-off floor, and reports two
complementary readings per axis:

  r_vs_real     correlation between the intact and the ablated prediction.
                Near 1.0 means the axis is doing nothing; near 0 means the
                output depends on it strongly. This is the quantity the old
                `ablation_r` estimated, now per axis.

  pearson_dev   the actual held-out metric under that ablation. This is the more
                decision-relevant number: an axis whose removal barely moves the
                metric is not contributing predictive skill, however much it
                moves the raw output.

Reading the table
-----------------
* If `ds_shuffle` barely changes pearson_dev, dataset identity was not carrying
  the score. If it collapses it, a large part of the reported skill is
  dataset-level commonality rather than perturbation-specific biology -- which is
  the same class of artefact as the P2 shared-response shortcut.
* If `is_tgt_off` alone collapses pearson_dev, the model is largely predicting
  "the perturbed gene goes down", which is true but nearly free.
* `all_off` is the floor: whatever pearson_dev survives there is what the model
  achieves with NO perturbation information at all. Any honest claim about
  conditioning is the gap between the intact model and that floor, not the gap
  from zero.

Usage
-----
    python src/maprna_p3/ablate_axes.py \
        --ckpt p3_out/ckpt_p3_best_dev.pt \
        --data_dirs <h5ad> [...] --esm_table <esm2.pt> \
        --split_by target_gene \
        [--string_neighbors string_neighbors.npy] \
        [--rna_encoder_ckpt rna_enc.pt --fasta gene_transcripts.fa] \
        --out_json ablation_report.json

`--split_by` and `--data_dirs` MUST match the training run, or the
reconstructed test split is not the model's test split.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "maprna_p1"))

from train_p3 import build_dev_data, make_rna_emb_lookup  # noqa: E402
from model_dev import DeviationModel, load_esm_matrix, load_dev_state  # noqa: E402
from eval_metrics import (  # noqa: E402
    bootstrap_ci_per_item, per_item_correlation, pooled_correlation, top_k_overlap)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from provenance import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--esm_table", required=True)
    p.add_argument("--string_neighbors", default="")
    p.add_argument("--rna_encoder_ckpt", default="")
    p.add_argument("--fasta", default="")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--hvg_from", choices=("train", "all"), default="train",
                   help="which perturbations the response panel is ranked on. Must "
                        "match the run being compared against: the panel is part of "
                        "the protocol, not a detail (docs/ERRATA.md E15).")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--split_by", default="target_gene",
                   choices=["target_gene", "pert"],
                   help="MUST match the training run")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--out_json", default="")
    return p.parse_args()


def main():
    args = parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_dev_data(args)

    rna_enc = None
    if args.rna_encoder_ckpt:
        data["rna_embs"] = make_rna_emb_lookup(args, data)

    esm = load_esm_matrix(args.esm_table)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_ds = ck["model_state_dict"]["ds_emb.weight"].shape[0]
    has_rna = any(k.startswith("proj_r.") for k in ck["model_state_dict"])
    if has_rna and not args.rna_encoder_ckpt:
        raise SystemExit(
            "this checkpoint has an RNA axis (proj_r.* present) but no "
            "--rna_encoder_ckpt was given; the RNA-axis ablation would be "
            "indistinguishable from simply never supplying the embedding.")
    model = DeviationModel(esm, data["hvg_rows"], n_ds=n_ds,
                           rna_encoder=rna_enc).to(dev).eval()
    load_dev_state(model, ck)
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), dev)
        print(f"[abl] STRING neighbour table loaded: "
              f"{tuple(model.neighbor_table.shape)}", flush=True)
    else:
        print("[abl] no STRING table -> is_neighbor is identically false, so the "
              "is_nb_shuffle row is uninformative for this run.", flush=True)

    items = data["test_items"]
    n = len(items)
    rows = torch.tensor(data["rows_te"], dtype=torch.long)
    ds_idx = torch.tensor([di for di, _ in items], dtype=torch.long)
    cf = torch.from_numpy(data["ctrl_feat_all"][ds_idx.numpy()])
    pe = torch.from_numpy(data["pert_expr_te"])
    rna = data["rna_embs"][len(data["train_items"]):] if model.proj_r is not None else None
    true_dev, mask = data["dev_te"], data["mask_te"]

    rng = np.random.default_rng(args.seed)
    perm = torch.from_numpy(rng.permutation(n))
    # A permutation with no fixed points, so that no perturbation keeps its own
    # value under a "shuffle" ablation (with a plain permutation ~1/e of a large
    # sample would, weakening the ablation).
    if n > 1:
        for _ in range(64):
            if not (perm == torch.arange(n)).any():
                break
            perm = torch.from_numpy(rng.permutation(n))

    is_nb_true, is_tgt_true = model.indicator_features(rows)

    def predict(esm_ov=None, rna_in="real", tgt=None, nb=None, ds=None):
        outs = []
        for lo in range(0, n, args.batch):
            hi = min(lo + args.batch, n)
            r = None
            if rna is not None and rna_in == "real":
                r = rna[lo:hi].to(dev)
            kw = {}
            if esm_ov is not None:
                kw["pert_esm_override"] = esm_ov[lo:hi].to(dev)
            if tgt is not None:
                kw["is_tgt_override"] = tgt[lo:hi]
            if nb is not None:
                kw["is_nb_override"] = nb[lo:hi]
            d_i = (ds if ds is not None else ds_idx)[lo:hi]
            with torch.no_grad():
                outs.append(model(rows[lo:hi].to(dev), d_i.to(dev), cf[lo:hi].to(dev),
                                  rna_emb=r, pert_ctrl_expr=pe[lo:hi].to(dev),
                                  **kw).cpu().numpy())
        return np.concatenate(outs)

    zero_esm = torch.zeros_like(model.esm_table[rows].cpu())
    shuf_esm = model.esm_table[rows[perm]].cpu()

    ABLATIONS = {
        "intact": dict(),
        "esm_zero": dict(esm_ov=zero_esm),
        "esm_shuffle": dict(esm_ov=shuf_esm),
        "rna_off": dict(rna_in="off"),
        "is_tgt_off": dict(tgt=torch.zeros_like(is_tgt_true)),
        "is_tgt_shuffle": dict(tgt=is_tgt_true[perm]),
        "is_nb_off": dict(nb=torch.zeros_like(is_nb_true)),
        "is_nb_shuffle": dict(nb=is_nb_true[perm]),
        "ds_shuffle": dict(ds=ds_idx[perm]),
        "all_off": dict(esm_ov=zero_esm, rna_in="off",
                        tgt=torch.zeros_like(is_tgt_true),
                        nb=torch.zeros_like(is_nb_true)),
    }

    real = predict()
    report = {}
    print(f"\nPer-axis conditioning ablation | n_test={n} "
          f"measured_fraction={mask.mean():.3f} split_by={args.split_by}\n", flush=True)
    print(f"{'ablation':16} {'r_vs_real':>10} {'pearson_dev':>12} "
          f"{'delta':>8} {'top50_dev':>10}", flush=True)
    base_pd = per_item_correlation(true_dev, real, mask)
    for name, kw in ABLATIONS.items():
        pred = real if name == "intact" else predict(**kw)
        a, b = real.ravel(), pred.ravel()
        r = (float(np.corrcoef(a, b)[0, 1])
             if a.std() > 1e-12 and b.std() > 1e-12 else float("nan"))
        pd_ = per_item_correlation(true_dev, pred, mask)
        report[name] = dict(
            r_vs_real=r, pearson_dev=pd_, delta_vs_intact=pd_ - base_pd,
            pearson_dev_pooled=pooled_correlation(true_dev, pred, mask),
            top50_dev=top_k_overlap(true_dev, pred, k=50, mask=mask))
        if name in ("intact", "all_off"):
            lo, hi = bootstrap_ci_per_item(true_dev, pred, mask, n_boot=500,
                                           seed=args.seed)
            report[name]["pearson_dev_ci95"] = [lo, hi]
        print(f"{name:16} {r:10.4f} {pd_:12.4f} "
              f"{pd_ - base_pd:+8.4f} {report[name]['top50_dev']:10.4f}", flush=True)

    floor = report["all_off"]["pearson_dev"]
    intact = report["intact"]["pearson_dev"]
    report["interpretation"] = dict(
        intact_pearson_dev=intact,
        no_conditioning_floor=floor,
        conditioning_gain=intact - floor,
        note=("The defensible claim about conditioning is the gap between the "
              "intact model and the all_off floor, not the intact value itself. "
              "A large ds_shuffle delta means dataset-level commonality is "
              "carrying part of the score, which is the same class of artefact "
              "as the P2 shared-response shortcut."))
    print(f"\nintact {intact:.4f} - no-conditioning floor {floor:.4f} "
          f"= conditioning gain {intact - floor:+.4f}", flush=True)
    if intact - floor < 0.05:
        print("WARNING: the model achieves nearly its full score with NO "
              "perturbation information. Conditioning is not contributing.",
              flush=True)

    if args.out_json:
        write_json(args.out_json,
                   dict(report=report, n_test=n, split_by=args.split_by,
                        measured_fraction=float(mask.mean()),
                        ckpt=os.path.abspath(args.ckpt)), args=args)
        print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
