#!/usr/bin/env python3
"""Inject STRING network context into the gene embedding table itself.

Why here, and not as a model channel
------------------------------------
Two measurements on real data (docs/RESULTS_REAL.md) point at this design:

1. The per-axis ablation shows the target gene's ESM2 vector is the **only** live
   conditioning channel -- `is_target`, `is_neighbor` and the RNA axis leave the
   prediction bit-identical or near enough. Network information therefore has to
   arrive *through* that vector to be used at all.
2. The `is_neighbor` indicator channel bought +0.0005, an order of magnitude below
   the seed noise, and it could not have bought more: a sparse binary partner
   flag over the response panel has support ~ degree/gene-universe ~ 0.2% of the
   scored matrix, near-independent of panel size. A prior confined to 0.2% of
   cells cannot move a panel-wide correlation.

So the network signal is folded into the embedding, where it is dense: every gene
gets a summary of its neighbourhood, and every gene's vector is affected.

Fairness, which is the whole point of doing it this way
------------------------------------------------------
This writes a **table**, not a model change. The same table is handed to the head,
to `ridge_esm2` and to `knn_esm2`, so all three see identical features and the
comparison isolates architecture rather than input. The previous attempt put
network information in a channel only the head could read; had it worked, the gain
would have been indistinguishable from simply giving the head more information --
the exact error this repository's errata document.

Modes
-----
concat  [own || mean(partners)], doubling the dimension. Keeps the original vector
        exactly, so nothing is destroyed and any gain is additive information.
blend   (1-alpha)*own + alpha*mean(partners), same dimension. Cheaper, but it
        mixes the self signal away, so a loss is ambiguous between "the network is
        unhelpful" and "the self vector was damaged".

Genes with no partners get a zero neighbourhood block, which makes "no network
information" a state the model can distinguish rather than a silent imputation.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esm_table", required=True)
    p.add_argument("--neighbors", required=True,
                   help="[V, K] int table from build_string_neighbors.py, built "
                        "against THIS esm_table's row order")
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("concat", "blend"), default="concat")
    p.add_argument("--alpha", type=float, default=0.3,
                   help="blend mode only: weight on the neighbourhood mean")
    p.add_argument("--normalise_block", action="store_true",
                   help="rescale the neighbourhood block to the same mean L2 norm "
                        "as the self block. Averaging shrinks a vector's norm, so "
                        "without this the network half enters at a smaller scale "
                        "than the self half -- which a linear model can absorb but "
                        "which biases early training of the head.")
    return p.parse_args()


def main():
    args = parse_args()
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab)
    V = len(symbols)
    X = torch.stack([tab[s] for s in symbols]).to(torch.float32).numpy()
    nb = np.load(args.neighbors)
    if nb.shape[0] != V:
        raise SystemExit(
            f"row-count mismatch: the embedding table has {V} rows and the "
            f"neighbour table has {nb.shape[0]}. Both are indexed by the same row "
            f"id, so mixing tables built against different symbol orders would "
            f"silently scramble the graph. Rebuild the neighbour table against "
            f"this embedding table.")

    # neighbourhood mean, zero where a gene has no partners
    valid = nb >= 0
    deg = valid.sum(axis=1)
    nbmean = np.zeros_like(X)
    idx = np.where(deg > 0)[0]
    for r in idx:
        nbmean[r] = X[nb[r, valid[r]]].mean(axis=0)
    n_iso = int((deg == 0).sum())
    print(f"[smooth] {V} genes | {V - n_iso} with partners | {n_iso} isolated "
          f"(zero neighbourhood block)", flush=True)

    self_norm = float(np.linalg.norm(X, axis=1).mean())
    nb_norm = float(np.linalg.norm(nbmean[idx], axis=1).mean()) if len(idx) else 0.0
    print(f"[smooth] mean L2 norm: self {self_norm:.3f} | neighbourhood "
          f"{nb_norm:.3f} (averaging shrinks it)", flush=True)
    scaled = False
    if args.normalise_block and nb_norm > 1e-8:
        nbmean *= (self_norm / nb_norm)
        scaled = True
        print(f"[smooth] neighbourhood block rescaled to match the self norm",
              flush=True)

    if args.mode == "concat":
        Y = np.concatenate([X, nbmean], axis=1)
    else:
        Y = (1.0 - args.alpha) * X + args.alpha * nbmean

    out = {s: torch.from_numpy(Y[i].copy()) for i, s in enumerate(symbols)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"[smooth] wrote {args.out}: {V} symbols x {Y.shape[1]} "
          f"(was {X.shape[1]}, mode={args.mode})", flush=True)
    write_json(str(Path(args.out).with_suffix(".provenance.json")), dict(
        out=os.path.abspath(args.out), n_symbols=V,
        dim_in=int(X.shape[1]), dim_out=int(Y.shape[1]), mode=args.mode,
        alpha=(args.alpha if args.mode == "blend" else None),
        source_esm_table=os.path.abspath(args.esm_table),
        neighbors=os.path.abspath(args.neighbors),
        genes_with_partners=int(V - n_iso), genes_isolated=n_iso,
        mean_norm_self=self_norm, mean_norm_neighbourhood=nb_norm,
        neighbourhood_block_rescaled=scaled,
        isolated_gene_policy="zero neighbourhood block, so 'no network "
                             "information' is a distinguishable state rather than "
                             "a silent imputation",
        fairness_note="This is a feature table, not a model change. Hand it to the "
                      "head AND to ridge_esm2/knn_esm2 so every method sees "
                      "identical inputs; otherwise a gain credits architecture "
                      "with what is really extra information.",
    ), args=args)


if __name__ == "__main__":
    main()
