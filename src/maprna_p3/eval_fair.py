"""Fair-comparison evaluation for the expanded response head.

Loads ckpt_p3_best_dev.pt (ep9 of the expanded run), rebuilds the same
deterministic split, and reports pearson_dev stratified by source dataset,
plus the 'legacy-domain' subset (test perts from adamson/norman/replogle)
as the distribution-level fair comparison against the old 0.3079 baseline.

Run on GPU host:  python $BASE/src/eval_fair.py --ckpt $BASE/p3_expanded/ckpt_p3_best_dev.pt ...
(same data args as train_p3)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_p3 import build_dev_data, make_rna_emb_lookup  # noqa: E402
from model_dev import DeviationModel  # noqa: E402

DS_NAMES = ["adamson", "norman", "replogle_ess", "gwps", "jurkat", "hepg2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--esm_table", required=True)
    p.add_argument("--string_neighbors", default="")
    p.add_argument("--rna_encoder_ckpt", default="")
    p.add_argument("--fasta", default="")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--split_by", type=str, default="target_gene",
                   choices=["target_gene", "pert"],
                   help="MUST match the value used for training, otherwise the "
                        "reconstructed test split is not the model's test split")
    p.add_argument("--out_json", default="",
                   help="optional path to write the stratified report as JSON")
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_dev_data(args)
    if args.rna_encoder_ckpt:
        data["rna_embs"] = make_rna_emb_lookup(args, data)
        print(f"[eval] rna embeddings: {len(data['rna_embs'])} perts", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Consistency guard: the rebuilt HVG panel / common_fc must be identical
    # to the ones the ckpt was trained with, otherwise the first-layer weights
    # do not line up with the genes (which shows up as a spurious regression:
    # overall pearson_dev collapsing to ~0.08).
    # Rule: the --data_dirs given to eval must be exactly the same as at
    # training time (same order).
    if "hvg_rows" in ck:
        if not np.array_equal(np.asarray(ck["hvg_rows"], dtype=np.int64),
                              np.asarray(data["hvg_rows"], dtype=np.int64)):
            raise RuntimeError(
                "the rebuilt HVG panel does not match the ckpt! the "
                "--data_dirs given to eval must be exactly the ones this ckpt "
                "was trained with (a 7-dataset ckpt, e.g. one with GSE293987 "
                "merged in, needs all 7 directories, in the same order). "
                f"ckpt panel n={len(ck['hvg_rows'])} vs "
                f"rebuilt n={len(data['hvg_rows'])}")
    if "common_fc" in ck:
        cf_ck = np.asarray(ck["common_fc"], dtype=np.float32)
        if not np.allclose(cf_ck, data["common_fc"], atol=1e-5):
            print("[warn] common_fc does not match the ckpt; the residual "
                  "targets have been overwritten with the ckpt value", flush=True)
            data["common_fc"] = cf_ck
            data["dev_tr"] = data["fc_tr"] - cf_ck
            data["dev_te"] = data["fc_te"] - cf_ck
    print(f"loaded ckpt: epoch {ck['epoch']} metrics={ {k: round(v, 4) for k, v in ck['metrics'].items() if isinstance(v, float)} }",
          flush=True)

    # rebuild model exactly as train_p3 does (incl. RNA encoder branch)
    rna_enc = None
    if args.rna_encoder_ckpt:
        from rna_encoder import RNAEncoder
        rna_enc = RNAEncoder(d_model=256, max_len=600)
        state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "rna_encoder" in state:
            state = state["rna_encoder"]    # Stage-A ckpt is a wrapper dict
        rna_enc.load_state_dict(state)
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    esm_matrix = torch.stack(list(tab.values())).float()
    # n_ds is derived automatically from the ckpt (the eval data_dirs need not
    # contain every training dataset, e.g. a 7-dataset ckpt after GSE293987 was
    # merged in can be evaluated with 6 directories)
    n_ds_ckpt = int(ck["model_state_dict"]["ds_emb.weight"].shape[0])
    print(f"[ckpt] n_ds={n_ds_ckpt} (derived; eval data n_ds={data['n_ds']})", flush=True)
    model = DeviationModel(esm_matrix, data["hvg_rows"], n_ds=n_ds_ckpt,
                           d_model=256, rna_encoder=rna_enc).to(dev).eval()
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), dev)
    # Pre-v4 checkpoints persist the frozen ESM2 table (~405 MB of the 428 MB
    # file). It is now a non-persistent buffer supplied at construction, so
    # drop the stale copy rather than failing on an unexpected key.
    _sd = {k: v for k, v in ck["model_state_dict"].items() if k != "esm_table"}
    model.load_state_dict(_sd)

    items = data["test_items"]
    ds_idx = torch.tensor([di for di, _ in items], dtype=torch.long)
    pr = torch.tensor(data["rows_te"], dtype=torch.long)
    cf = torch.from_numpy(data["ctrl_feat_all"][ds_idx.numpy()])
    rna = data["rna_embs"][len(data["train_items"]):] if model.proj_r is not None else None
    true_dev = data["dev_te"]
    mask = data["mask_te"]
    pe = torch.from_numpy(data["pert_expr_te"])

    # chunked inference.
    # pert_ctrl_expr must be passed here: train_p3.evaluate() passes it, so
    # omitting it (pre-v4 behaviour) silently evaluated a DIFFERENT model --
    # the v2-1a self-response gate was active during training and scoring but
    # inactive in this script.
    outs = []
    with torch.no_grad():
        for lo in range(0, pr.shape[0], args.batch):
            hi = min(lo + args.batch, pr.shape[0])
            kw = {"rna_emb": rna[lo:hi].to(dev)} if rna is not None else {}
            outs.append(model(pr[lo:hi].to(dev), ds_idx[lo:hi].to(dev),
                              cf[lo:hi].to(dev), pert_ctrl_expr=pe[lo:hi].to(dev),
                              **kw).cpu().numpy())
    pred_dev = np.concatenate(outs)

    # TWO ESTIMATORS, reported side by side and never subtracted from one another.
    #
    # `pooled` flattens the whole [n_pert, n_hvg] matrix into one correlation and
    # therefore also rewards getting the relative response AMPLITUDE of different
    # perturbations right. `per_pert` is the mean of within-perturbation
    # correlations and is what train_p3.evaluate() reports.
    #
    # Pre-v4, this script printed only the pooled value under the bare name
    # "pearson_dev", while train_p3 printed the per-perturbation value under the
    # same name. The published "0.31 -> 0.71" improvement subtracts one from the
    # other; on the same checkpoint they differ by roughly 1.5x, which is the
    # "in-training 0.2076 vs eval_fair 0.3141" discrepancy recorded as an open
    # backlog item in docs/reports/data_expansion_report.md.
    def pooled(a, b, sel=None, m=None):
        if sel is not None:
            a, b, m = a[sel], b[sel], m[sel]
        f = m.astype(bool)
        x, y = a[f], b[f]
        if x.size < 2 or x.std() < 1e-9 or y.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    def per_pert(a, b, sel=None, m=None):
        if sel is not None:
            a, b, m = a[sel], b[sel], m[sel]
        rs = []
        for i in range(len(a)):
            f = m[i].astype(bool)
            if f.sum() < 10:
                continue
            x, y = a[i][f], b[i][f]
            if x.std() > 1e-9 and y.std() > 1e-9:
                rs.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(rs)) if rs else float("nan")

    def line(label, sel=None):
        n = len(items) if sel is None else int(sel.sum())
        pp = per_pert(pred_dev, true_dev, sel, mask)
        po = pooled(pred_dev, true_dev, sel, mask)
        print(f"{label:34} n={n:5d}  pearson_dev_per_pert={pp:.4f}  "
              f"pearson_dev_pooled={po:.4f}", flush=True)
        return dict(n=n, pearson_dev_per_pert=pp, pearson_dev_pooled=po)

    report = {}
    print("\nEstimators: per_pert = mean of within-perturbation r (comparable to "
          "train_p3.evaluate); pooled = one r over the flattened matrix. "
          "They are NOT interchangeable.", flush=True)
    print(f"Masked to measured HVG columns only (measured fraction "
          f"{mask.mean():.3f}).\n", flush=True)
    report["overall"] = line("OVERALL")

    print("\n--- stratified by source dataset (test split) ---", flush=True)
    for d in range(data["n_ds"]):
        sel = np.array([di == d for di, _ in items])
        if sel.sum() == 0:
            continue
        name = DS_NAMES[d] if d < len(DS_NAMES) else f"ds{d}"
        report[name] = line(name, sel)

    print("", flush=True)
    report["legacy_domain"] = line(
        "LEGACY-DOMAIN (adamson/norman/repl_ess)",
        np.array([di in (0, 1, 2) for di, _ in items]))
    report["new_domain"] = line(
        "NEW-DOMAIN (gwps/jurkat/hepg2)",
        np.array([di in (3, 4, 5) for di, _ in items]))
    print("\nCaveat: this is a distribution-level comparison against the earlier "
          "1,843-perturbation baseline, not the identical 276-perturbation test "
          "set, and the HVG definition also changed between those runs. Compare "
          "per_pert with per_pert only.", flush=True)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(dict(report=report, measured_fraction=float(mask.mean()),
                           split_by=getattr(args, "split_by", "unknown")), f, indent=2)
        print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
