#!/usr/bin/env python3
"""Search graph-feature designs on INNER VALIDATION, never on the held-out set.

Why a sweep is worth doing at all
---------------------------------
Folding a gene's STRING neighbourhood into its embedding was the largest gain
measured in this project (docs/RESULTS_REAL.md): every method that reads the
embedding improved, and `train_mean`, which does not, did not move. That makes the
*feature table* the most productive thing to improve -- and because the table is
shared by the head and both ESM2 controls, improving it improves all of them
rather than handing one method an advantage.

Why ridge is the readout
------------------------
`ridge_esm2` is a closed-form solve: seconds, deterministic, no training budget,
no seed. On the graph features it scored +0.3370 against the head's +0.3346, so it
is not a weak proxy -- it is currently tied with the head. Sweeping with ridge and
verifying the winner with the head costs a tiny fraction of sweeping with the head,
and the asymmetry is stated rather than hidden: a variant that suits ridge need not
be the variant that suits the head, so the winner is a *candidate*, not a result.

The discipline this script exists to enforce
--------------------------------------------
Selecting a feature design by its held-out score and then reporting that score is
ERRATA E3 -- the defect that invalidated most of the withdrawn numbers, where the
reported epoch was chosen on the reported slice. So every number here is computed
on an inner-validation split carved out of the training items, and this script
**cannot** print a test score: it never touches `dev_te`. The winner has to be
re-run through train_p3.py, which scores the held-out set exactly once.
"""
import argparse
import importlib.util
import io
import contextlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "maprna_p3"),
                str(ROOT / "src" / "maprna_p1")]
from provenance import write_json  # noqa: E402
from baselines import baseline_ridge_esm2, baseline_knn_esm2  # noqa: E402
from eval_metrics import per_item_correlation  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esm_table", required=True, help="the PLAIN table, pre-graph")
    p.add_argument("--neighbors", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--n_hvg", type=int, default=500)
    p.add_argument("--min_cells", type=int, default=1)
    p.add_argument("--split_by", default="target_gene")
    p.add_argument("--hvg_from", default="train")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="a variant is ranked on its MEAN inner-val score over these "
                        "seeds; a single seed would pick whichever design suits one "
                        "split, which is how a sweep quietly cherry-picks")
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    return p.parse_args()


def load_train_p3():
    spec = importlib.util.spec_from_file_location(
        "tp", str(ROOT / "src" / "maprna_p3" / "train_p3.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["tp"] = m
    spec.loader.exec_module(m)
    return m


def neighbourhood_mean(X, nb, top_k):
    """Mean embedding of each gene's top-`top_k` partners; zeros if it has none.

    The neighbour table is score-sorted, so truncating its columns gives the
    top-k partners without rebuilding anything.
    """
    nbk = nb[:, :top_k]
    valid = nbk >= 0
    out = np.zeros_like(X)
    for r in np.where(valid.any(axis=1))[0]:
        out[r] = X[nbk[r, valid[r]]].mean(axis=0)
    return out


def make_variant(X, nbmean, mode, alpha, normalise):
    nm = nbmean
    if normalise:
        sn = np.linalg.norm(X, axis=1).mean()
        nn = np.linalg.norm(nbmean[np.abs(nbmean).sum(1) > 0], axis=1).mean()
        if nn > 1e-8:
            nm = nbmean * (sn / nn)
    if mode == "plain":
        return X
    if mode == "concat":
        return np.concatenate([X, nm], axis=1)
    return (1.0 - alpha) * X + alpha * nm


def main():
    args = parse_args()
    tp = load_train_p3()
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab)
    X = torch.stack([tab[s] for s in symbols]).to(torch.float32).numpy()
    nb = np.load(args.neighbors)
    if nb.shape[0] != X.shape[0]:
        raise SystemExit(f"row mismatch: table {X.shape[0]} vs neighbours "
                         f"{nb.shape[0]}; both are indexed by the same row id")

    # Every variant shares the symbol list, so the split and the HVG panel are
    # identical across variants and can be built once per seed.
    per_seed = {}
    for seed in args.seeds:
        a = Namespace(data_dirs=args.data_dirs, esm_table=args.esm_table,
                      n_hvg=args.n_hvg, test_frac=args.test_frac,
                      split_by=args.split_by, min_cells=args.min_cells, seed=seed,
                      inner_val_frac=args.inner_val_frac, hvg_from=args.hvg_from)
        with contextlib.redirect_stdout(io.StringIO()):
            d = tp.build_dev_data(a)
        iv = np.asarray(d["is_inner_val"], dtype=bool)
        per_seed[seed] = dict(
            rows_in=np.asarray(d["rows_tr"])[~iv], rows_va=np.asarray(d["rows_tr"])[iv],
            dev_in=d["dev_tr"][~iv], dev_va=d["dev_tr"][iv],
            mask_in=d["mask_tr"][~iv], mask_va=d["mask_tr"][iv])
        print(f"[sweep] seed {seed}: inner-train {int((~iv).sum())} / inner-val "
              f"{int(iv.sum())} items | panel {len(d['hvg_rows'])}", flush=True)

    VARIANTS = [dict(name="plain (no graph)", mode="plain", top_k=0, alpha=0.0, norm=False)]
    for k in (4, 8, 16, 32):
        VARIANTS.append(dict(name=f"concat k={k} norm", mode="concat", top_k=k,
                             alpha=0.0, norm=True))
    VARIANTS.append(dict(name="concat k=32 raw", mode="concat", top_k=32, alpha=0.0,
                         norm=False))
    for al in (0.2, 0.4):
        VARIANTS.append(dict(name=f"blend a={al} k=32", mode="blend", top_k=32,
                             alpha=al, norm=True))

    nbmean_cache = {}
    rows = []
    for v in VARIANTS:
        if v["top_k"] and v["top_k"] not in nbmean_cache:
            nbmean_cache[v["top_k"]] = neighbourhood_mean(X, nb, v["top_k"])
        nbm = nbmean_cache.get(v["top_k"], np.zeros_like(X))
        Xv = make_variant(X, nbm, v["mode"], v["alpha"], v["norm"])
        r_scores, k_scores = [], []
        for seed, d in per_seed.items():
            pr = baseline_ridge_esm2(d["dev_in"], Xv[d["rows_in"]], Xv[d["rows_va"]],
                                     mask_tr=d["mask_in"], alpha=args.ridge_alpha)
            r_scores.append(per_item_correlation(d["dev_va"], pr, d["mask_va"]))
            pk = baseline_knn_esm2(d["dev_in"], Xv[d["rows_in"]], Xv[d["rows_va"]],
                                   mask_tr=d["mask_in"], k=10)
            k_scores.append(per_item_correlation(d["dev_va"], pk, d["mask_va"]))
        rows.append(dict(**{k: v[k] for k in ("name", "mode", "top_k", "alpha", "norm")},
                         dim=int(Xv.shape[1]),
                         ridge_inner_val=float(np.mean(r_scores)),
                         ridge_per_seed=[float(x) for x in r_scores],
                         knn_inner_val=float(np.mean(k_scores)),
                         knn_per_seed=[float(x) for x in k_scores]))
        print(f"[sweep] {v['name']:22s} dim {Xv.shape[1]:5d}  ridge(inner-val) "
              f"{np.mean(r_scores):+.4f}  knn {np.mean(k_scores):+.4f}", flush=True)

    rows.sort(key=lambda r: -r["ridge_inner_val"])
    base = next(r for r in rows if r["mode"] == "plain")
    print("\n=== ranked on INNER-VAL ridge (no held-out number was computed) ===",
          flush=True)
    for r in rows:
        d = r["ridge_inner_val"] - base["ridge_inner_val"]
        print(f"  {r['name']:22s} {r['ridge_inner_val']:+.4f}  "
              f"({d:+.4f} vs plain)  knn {r['knn_inner_val']:+.4f}", flush=True)
    win = rows[0]
    print(f"\nCandidate: {win['name']}. This is a CANDIDATE, not a result -- it was "
          f"chosen on inner validation and must be scored once on the held-out set "
          f"by train_p3.py. Ridge chose it; the head may not agree.", flush=True)
    write_json(args.out_json, dict(
        ranked_on="inner-validation ridge, mean over seeds " + str(args.seeds),
        candidate=win["name"], variants=rows, plain_reference=base["ridge_inner_val"],
        esm_table=os.path.abspath(args.esm_table),
        neighbors=os.path.abspath(args.neighbors),
        discipline="No held-out score is computed anywhere in this script; dev_te is "
                   "never read. Selecting a design on the slice it is then reported "
                   "on is ERRATA E3, the defect behind most of the withdrawn numbers.",
        caveat="Ridge is the readout because it is closed-form and currently tied "
               "with the head on these features. A design that suits ridge need not "
               "suit the head, so the winner is a candidate to verify, not a result.",
    ), args=args)


if __name__ == "__main__":
    main()
