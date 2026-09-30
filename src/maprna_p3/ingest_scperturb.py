#!/usr/bin/env python3
"""Convert an scPerturb h5ad into the layout ds_knockdown.load_kd_datasets reads.

Why an adapter is needed
------------------------
Every number in this repository was produced from GEARS' own perturb_processed.h5ad
files, which are hosted on dataverse.harvard.edu. That host is refused by this
environment's egress policy, so the same experiments have to come from scPerturb
(Peidli et al. 2024) on Zenodo, which packages them differently:

  scPerturb                         what the loader expects
  ------------------------------    ---------------------------------------
  obs['perturbation'] = 'MYC'       obs['condition'] = 'MYC+ctrl'
  obs['perturbation'] = 'control'   obs['condition'] = 'ctrl', obs['control'] = 1
  X = raw UMI counts                X = log1p(counts / total * target_sum)
  var_names = gene symbols          var['gene_name'] = gene symbols

The normalisation is the part that cannot be skipped or reordered. The loader
takes plain means of X, and a mean of raw counts is a library-size-weighted
average, so a perturbation that happened to be sequenced deeper would get a
larger fold change than one that was not. Normalising per cell BEFORE the
pseudobulk mean is what makes the fold change a fold change.

What it writes
--------------
One pseudobulk row per surviving perturbation (the mean over all of that
perturbation's cells, computed streaming at full precision) plus up to --n_ctrl
real control cells. The loader pseudobulks perturbations itself, and a mean over
one row is that row, so this is the same quantity; the control rows stay
individual cells because the loader uses them as a pool.

One consequence has to be handled explicitly: the loader's --min_cells gate
counts rows in the file, and after aggregation every perturbation has exactly one
row. So this script applies the cell-count gate itself, on the true counts, and
records the threshold it used. Runs on its output must therefore pass
--min_cells 1; anything else would re-filter on a number that no longer means
cells.

Memory
------
X is read in row blocks and never held whole: ReplogleWeissman2022_rpe1 is a dense
247914 x 8749 float32 matrix, 8.7 GB, on a 15 GB host that also has to hold a
model.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from provenance import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True, help="input scPerturb h5ad")
    p.add_argument("--out", required=True, help="output h5ad in the loader's layout")
    p.add_argument("--min_cells", type=int, default=20,
                   help="drop a perturbation with fewer real cells than this. Applied "
                        "here, on true cell counts, because the pseudobulk output no "
                        "longer carries them (see the module docstring)")
    p.add_argument("--n_ctrl", type=int, default=3000,
                   help="control cells to carry over; the loader caps its own pool at "
                        "3000, so a larger number here is only wasted disk")
    p.add_argument("--target_sum", type=float, default=1e4,
                   help="per-cell library size the counts are scaled to before log1p")
    p.add_argument("--block", type=int, default=4096, help="rows read per block")
    p.add_argument("--seed", type=int, default=0, help="control-cell subsample")
    p.add_argument("--max_load_gb", type=float, default=4.0,
                   help="a CSC matrix has to be loaded whole rather than streamed; "
                        "refuse instead if that would need more than this")
    return p.parse_args()


def main():
    args = parse_args()
    import anndata as ad
    import h5py
    import pandas as pd
    import scipy.sparse as sp

    a = ad.read_h5ad(args.h5ad, backed="r")
    n_cells, n_genes_in = a.shape
    obs = a.obs
    if "perturbation" not in obs.columns:
        raise SystemExit(f"{args.h5ad}: no obs['perturbation']; not an scPerturb file")
    pert = obs["perturbation"].astype(str).to_numpy()
    syms_in = np.array([str(s) for s in a.var_names])

    # --- refuse rather than guess at combinations -------------------------------
    # scPerturb encodes multi-gene perturbations differently per dataset. Guessing
    # the separator would silently turn one combination into one wrong single-gene
    # label, so anything but single perturbations stops here.
    if "nperts" in obs.columns:
        nper = pd.to_numeric(obs["nperts"], errors="coerce")
        bad = int((nper.fillna(0) > 1).sum())
        if bad:
            raise SystemExit(
                f"{args.h5ad}: {bad} cells have nperts > 1. This adapter only handles "
                "single-gene perturbations; the combination separator differs per "
                "scPerturb dataset and guessing it would mislabel them.")

    ctrl_mask = np.char.lower(pert.astype(str)) == "control"
    if "nperts" in obs.columns:
        n0 = pd.to_numeric(obs["nperts"], errors="coerce").fillna(-1).to_numpy() == 0
        dis = int((ctrl_mask != n0).sum())
        if dis:
            print(f"[ingest] warning: {dis} cells disagree between "
                  f"perturbation=='control' and nperts==0; using the label", flush=True)
    if ctrl_mask.sum() == 0:
        raise SystemExit(
            f"{args.h5ad}: no cell has perturbation == 'control'. Without an "
            "unambiguous control population every fold change would be wrong, and "
            "picking a control guide by name is a guess. Refusing.")

    # --- collapse duplicated gene symbols --------------------------------------
    # Two columns with the same symbol both map to the same ESM2 row downstream,
    # where a dict would silently keep whichever came last. Summing raw counts is
    # the meaningful merge, and it has to happen before normalisation.
    uniq_syms, inv = np.unique(syms_in, return_inverse=True)
    n_dup = len(syms_in) - len(uniq_syms)
    n_genes = len(uniq_syms)
    collapse = sp.csr_matrix(
        (np.ones(len(inv), dtype=np.float32), (inv, np.arange(len(inv)))),
        shape=(n_genes, len(syms_in))) if n_dup else None
    print(f"[ingest] {args.h5ad}", flush=True)
    print(f"[ingest] cells={n_cells} genes={n_genes_in} -> {n_genes} unique symbols "
          f"({n_dup} duplicate columns summed)", flush=True)

    # --- conditions -------------------------------------------------------------
    cond = np.array(["ctrl" if c else f"{g}+ctrl" for g, c in zip(pert, ctrl_mask)])
    pert_conds, counts = np.unique(cond[~ctrl_mask], return_counts=True)
    keep = counts >= args.min_cells
    kept_conds = pert_conds[keep]
    print(f"[ingest] perturbations: {len(pert_conds)} -> {int(keep.sum())} with "
          f">= {args.min_cells} cells (dropped {int((~keep).sum())}); "
          f"cells/pert median {int(np.median(counts))}", flush=True)
    cond_to_k = {c: k for k, c in enumerate(kept_conds)}

    rng = np.random.default_rng(args.seed)
    ctrl_idx = np.where(ctrl_mask)[0]
    if len(ctrl_idx) > args.n_ctrl:
        ctrl_idx = np.sort(rng.choice(ctrl_idx, args.n_ctrl, replace=False))
    ctrl_pos = {int(i): j for j, i in enumerate(ctrl_idx)}
    print(f"[ingest] control cells: {int(ctrl_mask.sum())} -> {len(ctrl_idx)} kept",
          flush=True)

    # --- stream ----------------------------------------------------------------
    acc = np.zeros((len(kept_conds), n_genes), dtype=np.float64)
    n_acc = np.zeros(len(kept_conds), dtype=np.int64)
    ctrl_X = np.zeros((len(ctrl_idx), n_genes), dtype=np.float32)
    n_empty = 0
    with h5py.File(args.h5ad, "r") as h:
        Xh = h["X"]
        dense = isinstance(Xh, h5py.Dataset)
        if not dense:
            enc = dict(Xh.attrs).get("encoding-type", "")
            if enc not in ("csr_matrix", "csc_matrix"):
                raise SystemExit(f"unsupported X encoding {enc!r}")
            if enc == "csc_matrix":
                # Row blocks of a CSC matrix touch every column block, so streaming
                # it is slower than loading it. Load it whole and transpose -- but
                # only after checking it fits, because silently loading a matrix
                # that does not fit is how this script got OOM-killed once already.
                nnz = int(Xh["data"].shape[0])
                need_gb = (nnz * 8 + (len(syms_in) + 1) * 8) * 2 / 1e9
                if need_gb > args.max_load_gb:
                    raise SystemExit(
                        f"X is CSC with {nnz} non-zeros, about {need_gb:.1f} GB to "
                        f"load and convert, over --max_load_gb {args.max_load_gb}. "
                        "Convert the file to CSR first rather than risking the "
                        "OOM killer.")
                print(f"[ingest] X is CSC ({nnz} non-zeros, ~{need_gb:.1f} GB): "
                      f"loading whole and converting to CSR", flush=True)
                csr = sp.csc_matrix(
                    (np.asarray(Xh["data"][:], dtype=np.float32),
                     np.asarray(Xh["indices"][:]), np.asarray(Xh["indptr"][:])),
                    shape=(n_cells, len(syms_in))).tocsr()
                data, indices, indptr = csr.data, csr.indices, csr.indptr
            else:
                data, indices, indptr = Xh["data"], Xh["indices"], Xh["indptr"]
        for lo in range(0, n_cells, args.block):
            hi = min(lo + args.block, n_cells)
            if dense:
                blk = np.asarray(Xh[lo:hi], dtype=np.float32)
            else:
                p0, p1 = int(indptr[lo]), int(indptr[hi])
                blk = sp.csr_matrix(
                    (np.asarray(data[p0:p1], dtype=np.float32),
                     np.asarray(indices[p0:p1]),
                     np.asarray(indptr[lo:hi + 1]) - p0),
                    shape=(hi - lo, len(syms_in))).toarray()
            if collapse is not None:
                blk = (collapse @ blk.T).T
            tot = blk.sum(axis=1, keepdims=True)
            n_empty += int((tot[:, 0] <= 0).sum())
            blk = np.log1p(blk / np.maximum(tot, 1e-12) * args.target_sum)
            blk[tot[:, 0] <= 0] = 0.0          # a cell with no counts has no profile
            for r in range(hi - lo):
                i = lo + r
                if ctrl_mask[i]:
                    j = ctrl_pos.get(i)
                    if j is not None:
                        ctrl_X[j] = blk[r]
                else:
                    k = cond_to_k.get(cond[i])
                    if k is not None:
                        acc[k] += blk[r]
                        n_acc[k] += 1
            if (lo // args.block) % 10 == 0:
                print(f"[ingest]   {hi}/{n_cells} cells", flush=True)
    if n_empty:
        print(f"[ingest] warning: {n_empty} cells had zero total counts and "
              f"contribute a zero profile", flush=True)
    assert (n_acc == counts[keep]).all(), "streamed cell counts disagree with the tally"
    bulk = (acc / n_acc[:, None]).astype(np.float32)

    # --- write -----------------------------------------------------------------
    X_out = np.vstack([bulk, ctrl_X])
    obs_out = pd.DataFrame(dict(
        condition=np.concatenate([kept_conds, np.array(["ctrl"] * len(ctrl_idx))]),
        control=np.concatenate([np.zeros(len(kept_conds), int),
                                np.ones(len(ctrl_idx), int)]),
        n_cells=np.concatenate([n_acc, np.ones(len(ctrl_idx), int)]),
        is_pseudobulk=np.concatenate([np.ones(len(kept_conds), int),
                                      np.zeros(len(ctrl_idx), int)]),
    ))
    obs_out.index = [f"pb_{c}" for c in kept_conds] + [f"ctrl_{i}" for i in ctrl_idx]
    var_out = pd.DataFrame(dict(gene_name=uniq_syms), index=uniq_syms)
    out = ad.AnnData(X=X_out, obs=obs_out, var=var_out)
    out.uns["ingest"] = dict(
        source=os.path.abspath(args.h5ad), normalisation=f"log1p(counts/total*{args.target_sum:g})",
        min_cells=args.min_cells, note="perturbation rows are pseudobulk means; run "
                                       "train_p3.py with --min_cells 1")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out)
    write_json(str(Path(args.out).with_suffix(".provenance.json")), dict(
        source=os.path.abspath(args.h5ad), out=os.path.abspath(args.out),
        n_cells_in=int(n_cells), n_genes_in=int(n_genes_in), n_genes_out=int(n_genes),
        duplicate_symbol_columns_summed=int(n_dup),
        n_perturbations_in=int(len(pert_conds)), n_perturbations_out=int(len(kept_conds)),
        min_cells=args.min_cells, cells_per_pert_median=float(np.median(counts)),
        n_control_cells_available=int(ctrl_mask.sum()), n_control_cells_kept=len(ctrl_idx),
        n_cells_zero_counts=int(n_empty),
        normalisation=f"per cell: counts / total * {args.target_sum:g}, then log1p; "
                      "applied BEFORE the pseudobulk mean",
        downstream_requirement="train_p3.py --min_cells 1 (the cell-count gate ran here)",
    ), args=args)
    print(f"[ingest] wrote {args.out}: {X_out.shape[0]} rows "
          f"({len(kept_conds)} pseudobulk + {len(ctrl_idx)} ctrl) x {n_genes} genes",
          flush=True)


if __name__ == "__main__":
    main()
