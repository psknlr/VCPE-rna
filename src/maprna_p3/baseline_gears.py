#!/usr/bin/env python
"""GEARS as an external baseline, on VCPE's split, panel and metrics.

Why this script exists
----------------------
This repository quoted GEARS, scGPT, CPA and AIDO.RNA-Pert from their papers and
never ran any of them (docs/ERRATA.md E10). Quoting a number obtained under a
different split, a different gene space and a different estimator is not a
comparison. This runs GEARS for real and scores it the same way the VCPE head is
scored.

What makes the comparison fair, and what still does not
-------------------------------------------------------
Four things are forced to match, because each of them can move a number more than
the architectural difference being tested:

1. **The split.** GEARS is given VCPE's exact train/val/test condition sets via
   its `split='custom'` mechanism, so both models hold out the same
   perturbations. With `--split_by target_gene` that means the same held-out
   *genes*, which is the only split under which either model's number describes
   generalisation (E5).

2. **The predicted quantity.** VCPE predicts a residual: `dev = fc - common_fc`,
   where `fc` is the log fold-change against the dataset's control mean and
   `common_fc` is the *train-only* mean fold-change. GEARS predicts
   post-perturbation expression. Its output is therefore converted the same way:
   expression -> fc against the same control mean -> minus the same train-only
   `common_fc`. Comparing GEARS's raw expression correlation against VCPE's
   residual correlation would flatter GEARS enormously, because the shared
   response dominates raw expression -- which is the entire lesson of the P2
   erratum.

3. **The gene space and the mask.** Scored on VCPE's HVG panel only, with the
   same measured-column mask, so neither model is credited for columns its
   dataset never measured (E6).

4. **The estimator.** The same `per_item_correlation` / `pooled_correlation` /
   `top_k_overlap` functions, named explicitly (E4).

What is *not* controlled, and must be reported alongside any result:

* **Tuning effort.** GEARS is run at its defaults. VCPE's hyperparameters were
  chosen against this data over many runs. That asymmetry favours VCPE, and no
  amount of protocol alignment removes it -- the honest fix is a tuning budget
  for both, which this script does not attempt.
* **GEARS's own preprocessing.** `PertData` recomputes DE genes and its own
  perturbation graph. Those are part of GEARS and are left alone; disabling them
  would not be "GEARS".
* **Capacity.** GEARS at defaults is far larger than the 5.7M-parameter VCPE
  head. If VCPE wins, it wins as the smaller model; if it loses, the size
  difference is an explanation, not an excuse.

Usage
-----
    python src/maprna_p3/baseline_gears.py \
        --data_dirs data/adamson/perturb_processed.h5ad \
        --esm_table data/drive_weights/...ESM2.pt \
        --split_by target_gene --epochs 20 \
        --out_json gears_baseline.json

Single dataset at a time: GEARS's PertData takes one AnnData, so a multi-dataset
VCPE run has no single GEARS counterpart. Run it per dataset and compare against
`eval_fair.py`'s per-dataset rows.
"""
import argparse
import contextlib
import json
import os
import pickle
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "maprna_p1"))

from train_p3 import build_dev_data  # noqa: E402
from baseline_common import (  # noqa: E402
    align_to_panel, expression_to_dev, fairness_block, hvg_symbols,
    inner_split_items, intersect_masks, score)
from provenance import write_json  # noqa: E402


@contextlib.contextmanager
def _dense_X(pert_data, max_dense_gb):
    """Temporarily densify pert_data.adata.X, then restore the sparse matrix.

    GEARS is internally inconsistent about X's type on a current scipy/pandas,
    and the requirements alternate:

      * `data_utils.get_dropout_non_zero_genes` (inside `new_data_process`) calls
        `adata.X.toarray()`           -> X must be SPARSE
      * `GEARS.__init__` does
        `self.adata.X[self.adata.obs.condition == 'ctrl']`, indexing with a pandas
        boolean Series, which current scipy rejects for a sparse matrix
        ("'Series' object has no attribute 'nonzero'")
                                      -> X must be DENSE
      * `utils.get_coexpression_network_from_train` (inside `model_initialize`)
        calls `X_tr.toarray()`        -> X must be SPARSE again

    So the conversion cannot be done once up front; it has to be scoped to the
    constructor. This is a version incompatibility inside GEARS, not something
    this adapter introduces, and it is worth recording: a GEARS comparison is not
    reproducible on a current environment without either this workaround or a
    scipy old enough that sparse matrices accept a pandas boolean Series.
    Whichever was used belongs next to the reported number.
    """
    from scipy import sparse as sp
    original = pert_data.adata.X
    if sp.issparse(original):
        need_gb = pert_data.adata.n_obs * pert_data.adata.n_vars * 4 / 1e9
        if need_gb > max_dense_gb:
            raise SystemExit(
                f"GEARS needs a dense X for its constructor on this scipy "
                f"({need_gb:.1f} GB for {pert_data.adata.n_obs} x "
                f"{pert_data.adata.n_vars}, limit --max_dense_gb={max_dense_gb}). "
                f"Raise the limit, pass a pseudobulk-aggregated h5ad, or install a "
                f"scipy old enough to accept a pandas boolean Series.")
        print(f"[gears] densifying X ({need_gb:.3f} GB) for the GEARS constructor "
              f"only", flush=True)
        pert_data.adata.X = np.asarray(original.todense(), dtype=np.float32)
    try:
        yield
    finally:
        pert_data.adata.X = original


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="exactly one h5ad: GEARS's PertData takes a single AnnData")
    p.add_argument("--esm_table", required=True,
                   help="only used to rebuild VCPE's split and panel identically")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--hvg_from", choices=("train", "all"), default="train",
                   help="which perturbations the response panel is ranked on. Must "
                        "match the run being compared against: the panel is part of "
                        "the protocol, not a detail (docs/ERRATA.md E15).")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--split_by", default="target_gene",
                   choices=["target_gene", "pert"],
                   help="MUST match the VCPE run being compared against")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_max_genes", type=int, default=5000,
                   help="subset GEARS's training matrix to this many genes (0 = all). "
                        "GEARS builds one PyG object per cell over every gene, so a "
                        "33k-gene cell-level file exhausts a 15 GB host; it is also "
                        "not GEARS's own regime, which is about 5000 highly variable "
                        "genes. The genes VCPE scores are ALWAYS kept, whatever their "
                        "variance, so coverage of the panel stays complete and the "
                        "subsetting cannot quietly remove the genes being compared. "
                        "Reducing genes is preferred over reducing cells: fewer cells "
                        "would handicap GEARS, and a comparison that handicaps the "
                        "external baseline is worthless.")
    p.add_argument("--train_h5ad", default=None,
                   help="cell-level h5ad GEARS trains on. Default: the same file as "
                        "--data_dirs. GEARS is a cell-level model -- its own "
                        "differential-expression step calls scanpy's "
                        "rank_genes_groups per condition, which fails outright on a "
                        "pseudobulk file because a group of one sample has no "
                        "statistics. Pointing it at a cells export (ingest_scperturb "
                        "--mode cells) lets GEARS train the way it is published, "
                        "while the split, the gene panel and the scored quantity all "
                        "still come from the file VCPE reads. That asymmetry is "
                        "deliberate and is stated in the fairness block: handicapping "
                        "the external baseline would make the comparison worthless.")
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden_size", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gears_work_dir", default="",
                   help="scratch directory for PertData; a temporary one by default")
    p.add_argument("--max_dense_gb", type=float, default=8.0,
                   help="memory ceiling for the dense-X workaround GEARS "
                        "needs on current scipy; see the note in main()")
    p.add_argument("--runs", type=int, default=5,
                   help="repeat the whole train+predict this many times and report "
                        "the distribution. GEARS is NOT deterministic at a fixed "
                        "seed -- measured here, 7 identical invocations spanned "
                        "-0.023..+0.051 pearson_dev (sd 0.022, larger than the "
                        "mean itself). A single run is not a usable baseline.")
    p.add_argument("--out_json", default="")
    return p.parse_args()


def vcpe_split_to_gears(train_items, val_items, test_items):
    """VCPE (dataset_idx, condition) items -> GEARS's {'train','val','test'} dict.

    GEARS keys its split by condition string, so the dataset index is dropped --
    which is why this script takes a single dataset. `ctrl` is never a member of
    any split in either framework.
    """
    def conds(items):
        return sorted({c for _, c in items if c.lower() != "ctrl"})
    return {"train": conds(train_items), "val": conds(val_items),
            "test": conds(test_items)}


def align_to_hvg(pred_expr, gears_gene_names, hvg_symbols_):
    """Kept as a name for the tests; the implementation is shared with every
    other baseline in baseline_common.align_to_panel, so no baseline can drift
    into aligning genes differently from another."""
    return align_to_panel(pred_expr, gears_gene_names, hvg_symbols_)


def main():
    args = parse_args()
    if len(args.data_dirs) != 1:
        raise SystemExit(
            "GEARS's PertData takes a single AnnData, so pass exactly one h5ad "
            "and compare against eval_fair.py's row for that dataset. Got "
            f"{len(args.data_dirs)}.")

    # ---- 1. rebuild VCPE's split / panel / residual convention identically ----
    data = build_dev_data(args)
    hvg_rows = np.asarray(data["hvg_rows"])
    train_items, val_items, test_items = inner_split_items(data)
    print(f"[vcpe] split_by={args.split_by}: train {len(train_items)} / "
          f"val {len(val_items)} / test {len(test_items)} conditions", flush=True)

    # HVG row ids -> gene symbols, so GEARS's var_names can be aligned to them
    panel_symbols = hvg_symbols(data)
    n_named = sum(1 for s in panel_symbols if s)
    print(f"[vcpe] HVG panel: {len(hvg_rows)} rows, {n_named} with a symbol", flush=True)

    # ---- 2. hand GEARS that split verbatim ----
    work = args.gears_work_dir or tempfile.mkdtemp(prefix="gears_")
    os.makedirs(work, exist_ok=True)
    split_dict = vcpe_split_to_gears(train_items, val_items, test_items)
    split_fp = os.path.join(work, "vcpe_split.pkl")
    with open(split_fp, "wb") as f:
        pickle.dump(split_dict, f)
    print(f"[gears] custom split written: "
          f"{ {k: len(v) for k, v in split_dict.items()} }", flush=True)

    import anndata as ad
    from gears import GEARS, PertData

    train_h5ad = args.train_h5ad or args.data_dirs[0]
    adata = ad.read_h5ad(train_h5ad)
    if "gene_name" in adata.var.columns:          # GEARS keys on var_names
        adata.var_names = adata.var["gene_name"].astype(str)
        adata.var_names_make_unique()
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "cells"           # PertData requires the column

    # GEARS's differential-expression step needs more than one cell per condition.
    # Without this check it dies inside scanpy with a ValueError naming every
    # perturbation at once, which reads like a data problem rather than the wrong
    # input granularity.
    if args.train_max_genes and adata.n_vars > args.train_max_genes:
        import scipy.sparse as _sp
        panel = {s_ for s_ in panel_symbols if s_}
        names = np.asarray([str(v) for v in adata.var_names])
        must = np.isin(names, list(panel))
        X_ = adata.X
        # variance without densifying: E[x^2] - E[x]^2
        if _sp.issparse(X_):
            m1 = np.asarray(X_.mean(axis=0)).ravel()
            m2 = np.asarray(X_.multiply(X_).mean(axis=0)).ravel()
        else:
            m1 = np.asarray(X_).mean(axis=0)
            m2 = (np.asarray(X_) ** 2).mean(axis=0)
        var = np.maximum(m2 - m1 ** 2, 0.0)
        var[must] = np.inf                      # the scored panel is never dropped
        keep = np.sort(np.argsort(-var)[:max(args.train_max_genes, int(must.sum()))])
        kept_panel = int(np.isin(names[keep], list(panel)).sum())
        print(f"[gears] training matrix subset to {len(keep)}/{adata.n_vars} genes "
              f"({kept_panel}/{len(panel)} panel genes kept, all of them by "
              f"construction)", flush=True)
        assert kept_panel == len(panel & set(names)), "a scored gene was dropped"
        adata = adata[:, keep].copy()

    n_per_cond = adata.obs["condition"].astype(str).value_counts()
    thin = n_per_cond[n_per_cond < 2]
    if len(thin):
        raise SystemExit(
            f"{train_h5ad}: {len(thin)} of {len(n_per_cond)} conditions have a "
            "single row, so GEARS's rank_genes_groups step has no statistics to "
            "compute and fails. This is what a pseudobulk export looks like to "
            "GEARS. Build a cell-level file with\n"
            "  python src/maprna_p3/ingest_scperturb.py --mode cells ...\n"
            "and pass it as --train_h5ad, keeping --data_dirs on the pseudobulk "
            "file so the split, panel and scored quantity stay identical to the "
            "VCPE run.")

    # the comparison is only on the same split if both files hold the same
    # conditions; a mismatch would silently move the split
    vcpe_conds = {c for _, c in train_items + val_items + test_items}
    train_conds = set(n_per_cond.index) - {"ctrl"}
    missing = sorted(vcpe_conds - train_conds)
    if missing:
        raise SystemExit(
            f"{train_h5ad} is missing {len(missing)} of the {len(vcpe_conds)} "
            f"conditions VCPE splits on, e.g. {missing[:5]}. The two files must "
            "come from the same source and the same --min_cells, or the split is "
            "not shared.")
    extra = sorted(train_conds - vcpe_conds)
    if extra:
        print(f"[gears] note: {len(extra)} conditions in the training file are not "
              f"in VCPE's item set and are unused, e.g. {extra[:3]}", flush=True)

    pert_data = PertData(work)
    pert_data.new_data_process(dataset_name="vcpe_cmp", adata=adata)

    pert_data.prepare_split(split="custom", split_dict_path=split_fp)
    pert_data.get_dataloader(batch_size=args.batch_size,
                             test_batch_size=args.batch_size)

    # ---- 3. train GEARS at its defaults, repeatedly ----
    # GEARS(...) is the only step that needs a dense X, so densify for exactly
    # that call and restore the sparse matrix afterwards. See _dense_X.
    def train_once(run_idx):
        with _dense_X(pert_data, args.max_dense_gb):
            m = GEARS(pert_data, device=args.device)
        m.model_initialize(hidden_size=args.hidden_size)
        print(f"[gears] run {run_idx + 1}/{args.runs}: training {args.epochs} "
              f"epochs on {args.device} ...", flush=True)
        m.train(epochs=args.epochs)
        return m

    model = train_once(0)

    # ---- 4. predict the test conditions ----
    # GEARS can only predict perturbations present in its perturbation graph,
    # which is derived from a gene-set / GO resource rather than from the data.
    # A target gene can therefore be in the training split and still be
    # unpredictable, so report the coverage before scoring: a low number makes
    # any resulting metric a statement about a biased subset of the test set.
    test_genes = [c.split("+")[0] for _, c in test_items]
    in_graph = [g for g in test_genes if g in set(model.pert_list)]
    print(f"[gears] perturbation graph: {len(model.pert_list)} genes | test genes "
          f"in graph: {len(in_graph)}/{len(test_genes)}", flush=True)
    if not in_graph:
        raise SystemExit(
            f"None of the {len(test_genes)} held-out target genes is in GEARS's "
            f"perturbation graph, so GEARS cannot predict any of them and there is "
            f"nothing to compare. This is expected for synthetic or non-HGNC gene "
            f"names. Examples tried: {test_genes[:5]}. Use real gene symbols, or "
            f"pass a --gene_set_path that covers them.")

    gears_genes = list(pert_data.adata.var_names)
    ctrl = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    ctrl_mean_full = np.asarray(ctrl.X.mean(axis=0)).ravel()
    c_expr, c_found = align_to_panel(ctrl_mean_full, gears_genes, panel_symbols)

    def predict_all(m):
        """Predicted residuals + coverage for every held-out condition."""
        pred = np.zeros((len(test_items), len(hvg_rows)), dtype=np.float32)
        cover = np.zeros_like(pred, dtype=bool)
        missed = []
        for k, (_, cond) in enumerate(test_items):
            gene = cond.split("+")[0]
            try:
                # GEARS.predict returns (results_pred, results_logvar) only when
                # the uncertainty head is enabled and results_pred alone otherwise,
                # so unpacking a 2-tuple unconditionally fails on the default config.
                out = m.predict([[gene]])
                res = out[0] if isinstance(out, tuple) else out
            except Exception as e:
                missed.append((cond, f"{type(e).__name__}: {str(e)[:70]}"))
                continue
            expr = np.asarray(res[next(iter(res))]).ravel()
            p_expr, f1 = align_to_panel(expr, gears_genes, panel_symbols)
            # the same conversion VCPE's target uses: fc against the same control
            # mean, minus the same train-only common core
            pred[k] = expression_to_dev(p_expr, c_expr, data["common_fc"])
            cover[k] = f1 & c_found
        return pred, cover, missed

    # ---- 5. repeat, then score with VCPE's mask AND GEARS's coverage ----
    # Intersecting the two masks is the only honest choice: a column VCPE never
    # measured has no ground truth, and a column GEARS does not predict has no
    # prediction. Scoring either as "no change" would credit or penalise a model
    # for a gene it was never asked about.
    true_dev = data["dev_te"]
    per_run, skipped = [], []
    for r in range(args.runs):
        m = model if r == 0 else train_once(r)
        pred_dev, found_mask, missed = predict_all(m)
        skipped = missed
        mask = intersect_masks(data["mask_te"], found_mask)
        per_run.append(dict(run=r + 1, mask_fraction=float(mask.mean()),
                            n_scored=int(mask.any(axis=1).sum()),
                            **score(true_dev, pred_dev, mask, seed=args.seed + r)))
        print(f"[gears] run {r + 1}/{args.runs}: pearson_dev "
              f"{per_run[-1]['pearson_dev']:+.4f}", flush=True)
    if skipped:
        print(f"[gears] {len(skipped)} test conditions could not be predicted: "
              f"{skipped[:5]}", flush=True)

    vals = np.array([r["pearson_dev"] for r in per_run], dtype=float)
    finite = vals[np.isfinite(vals)]
    across = dict(
        n_runs=int(len(finite)),
        pearson_dev_mean=float(finite.mean()) if finite.size else float("nan"),
        pearson_dev_sd=(float(finite.std(ddof=1)) if finite.size > 1 else None),
        pearson_dev_min=float(finite.min()) if finite.size else float("nan"),
        pearson_dev_max=float(finite.max()) if finite.size else float("nan"),
        note=("GEARS is not deterministic at a fixed seed. Measured on this setup, "
              "7 identical invocations spanned -0.023..+0.051 pearson_dev with a "
              "sample sd of 0.022 -- larger than the mean itself. Quote the mean "
              "and spread across runs, never a single run."),
    )
    report = dict(
        n_test_conditions=int(len(test_items)),
        train_h5ad=os.path.abspath(train_h5ad),
        use_mask=True,  # scored through the intersected measured mask, unconditionally
        protocol=dict(data_dirs=[os.path.abspath(d) for d in args.data_dirs],
                      n_hvg=args.n_hvg, hvg_from=getattr(args, "hvg_from", "train"),
                      split_by=args.split_by, test_frac=args.test_frac,
                      min_cells=args.min_cells, seed=args.seed, use_mask=True,
                      esm_table=os.path.abspath(args.esm_table)),
        train_h5ad_is_cell_level=bool(args.train_h5ad),
        train_n_genes=int(adata.n_vars),
        train_n_cells=int(adata.n_obs),
        n_skipped_not_in_pert_graph=len(skipped),
        skipped=[c for c, _ in skipped],
        vcpe_measured_fraction=float(data["mask_te"].mean()),
        gears_per_run=per_run,
        gears_across_runs=across,
        config=dict(split_by=args.split_by, seed=args.seed, epochs=args.epochs,
                    hidden_size=args.hidden_size, device=args.device,
                    n_hvg=args.n_hvg, runs=args.runs),
        fairness=fairness_block(extra=[
            "GEARS's perturbation graph is GO-derived, so a target gene can be in "
            "the training split and still be unpredictable; the count of held-out "
            "genes present in the graph is reported above.",
            "GEARS is broken on a current scipy/pandas and required a dense-X "
            "workaround scoped to its constructor; see the note in _dense_X. A "
            "GEARS comparison is not reproducible today without that or an older "
            "scipy, and which was used belongs next to this number.",
            "GEARS does not reproduce its own result at a fixed seed, so its number "
            "here is a mean over --runs repetitions with the spread reported.",
        ] + ([
            "Input granularity differs on purpose. GEARS trained on individual "
            f"cells ({os.path.basename(train_h5ad)}) while the P3 head trains on "
            "the pseudobulk mean of the same cells; the split, the gene panel, the "
            "predicted quantity and the metric are identical. GEARS cannot run on "
            "pseudobulk at all -- its differential-expression step has no "
            "statistics for a group of one -- so the alternative was not a fairer "
            "comparison but no comparison.",
        ] if args.train_h5ad else [
            "GEARS was given the same pseudobulk file as the P3 head rather than "
            "the cell-level input it is published on, which is not the regime it "
            "was designed for.",
        ])),
    )

    print(f"\n=== GEARS on VCPE's split, panel and metrics "
          f"({across['n_runs']} runs) ===", flush=True)
    sd = across["pearson_dev_sd"]
    print(f"  pearson_dev  {across['pearson_dev_mean']:+.4f}"
          + (f" +/- {sd:.4f} (sample sd)" if sd is not None else "")
          + f"  range {across['pearson_dev_min']:+.4f}..{across['pearson_dev_max']:+.4f}",
          flush=True)
    print(f"  mask fraction {per_run[0]['mask_fraction']:.3f} | scored "
          f"{per_run[0]['n_scored']}/{report['n_test_conditions']} conditions",
          flush=True)
    print("\nCompare against the VCPE head's pearson_dev from the SAME "
          "--split_by / --seed / --n_hvg, and read the fairness notes before "
          "quoting either number.", flush=True)

    if args.out_json:
        write_json(args.out_json, report, args=args)
        print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
