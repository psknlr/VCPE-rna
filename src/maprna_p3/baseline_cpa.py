#!/usr/bin/env python
"""CPA as an external baseline, run in an isolated environment.

Why this is structured as three stages instead of one script
-----------------------------------------------------------
CPA cannot share an environment with this repository. `cpa-tools` pins
`torch<2.0.0` (through `scvi-tools==0.20.3`), while VCPE requires `torch>=2.1`.
Installing CPA into the VCPE environment silently downgraded torch from 2.x to
1.13.1 and broke the model code; pip states the conflict outright:

    cpa-tools 0.8.1 requires torch<2.0.0,>1.8.0, but you have torch 2.14.0

So a single-process adapter of the kind used for GEARS is impossible here. The
comparison is split at the process boundary, with the interface on disk:

  1. `export`  (VCPE env)  -- write the split, the gene panel, the control mean,
                              the train-only common core, and the held-out
                              residual targets to a directory.
  2. `run`     (CPA env)   -- read that directory, train CPA, write predicted
                              expression for the held-out conditions.
  3. `score`   (VCPE env)  -- read the predictions, convert them to VCPE's
                              residual, apply the mask, compute the metrics.

`--prepare_venv` creates the isolated environment. This structure is not CPA
overhead: any future baseline with conflicting pins plugs into the same three
stages, and stage 3 is shared with every other baseline via baseline_common, so
no baseline can drift into being scored differently (which is how ERRATA E1/E2
happened one level down).

What is controlled, and what is not
-----------------------------------
`baseline_common.fairness_block()` states both, and every report carries it.
Briefly: the split, the predicted quantity, the gene space and mask, and the
estimator are all forced to match. Tuning effort and model capacity are not, and
that asymmetry favours VCPE.

CPA's split is injected the clean way: CPA reads a `split` column from `obs` and
takes `train_split` / `valid_split` / `test_split` by name, so VCPE's own split is
written into that column rather than being approximated.

Usage
-----
    # once
    python src/maprna_p3/baseline_cpa.py --prepare_venv --venv .venv_cpa

    # 1) in the VCPE environment
    python src/maprna_p3/baseline_cpa.py export --work cpa_run \\
        --data_dirs data/adamson/perturb_processed.h5ad \\
        --esm_table data/drive_weights/...ESM2.pt --split_by target_gene

    # 2) in the CPA environment
    .venv_cpa/bin/python src/maprna_p3/baseline_cpa.py run --work cpa_run \\
        --max_epochs 100

    # 3) back in the VCPE environment
    python src/maprna_p3/baseline_cpa.py score --work cpa_run \\
        --out_json cpa_baseline.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "maprna_p1"))

# CPA's pins forbid torch>=2, so the `run` stage must not import anything from
# this repository that needs it. Only `export` and `score` touch VCPE code.
# Four separate walls stand between cpa-tools 0.8.1 and a current environment.
# Each was hit in turn while getting this to run, and all four are recorded in
# docs/ERRATA.md because together they are the real cost of a CPA comparison:
#   1. cpa-tools pins torch<2.0.0; this repository needs torch>=2.1. Installing
#      CPA alongside VCPE downgrades torch and breaks the model code -- hence the
#      separate venv and the three-stage structure of this script.
#   2. torch 1.13 does not import on Python 3.11+ (a dataclass rule tightened),
#      so the venv is built on Python 3.10.
#   3. cpa/_model.py has a stray `from tkinter import N` that it never uses,
#      making CPA unimportable without tkinter -- see _shim_tkinter.
#   4. scvi-tools 0.20.3 imports anndata's SparseDataset, which newer anndata
#      removed, so anndata has to be pinned back too.
#   5. scvi-tools 0.20.3 imports `jaxlib.xla_extension.Device`. That module no
#      longer exists, and jaxlib older than 0.4.14 is no longer distributed for
#      Python 3.10, so pinning back is not an option. Device was RENAMED rather
#      than removed (it is jaxlib.xla_client.Device now) and scvi uses it only as
#      a type annotation, so _shim_jaxlib re-exports it -- see that function.
# The pattern is the point: cpa-tools 0.8.1's transitive closure is unsatisfiable
# without pinning a good part of the scientific Python stack to 2023 versions.
# That is the real cost of the comparison and belongs in a methods section.
#   6. Its dependency stack still uses np.float_, removed in NumPy 2.0.
CPA_PINS = ["cpa-tools==0.8.1", "anndata<0.10", "scanpy<1.10", "numpy<2"]


# ---------------------------------------------------------------- venv --------

def _find_python(want="3.10"):
    """Locate an interpreter of the requested minor version, or None."""
    import shutil
    for name in (f"python{want}", f"/usr/bin/python{want}", f"/usr/local/bin/python{want}"):
        path = shutil.which(name) or (name if os.path.exists(name) else None)
        if path:
            return path
    return None

def prepare_venv(venv_dir, python=None):
    """Create an isolated environment for CPA and report what it pinned.

    Needs an interpreter CPA can actually run on. cpa-tools 0.8.1 pins
    torch<2.0.0, and torch 1.13 does not work on Python 3.11 or later -- it fails
    at import with `ValueError: mutable default ... use default_factory`, a
    dataclass rule that tightened in 3.11. So the venv is built on Python 3.10 by
    default rather than on whatever is running this script.
    """
    base = python or _find_python("3.10") or sys.executable
    print(f"[venv] creating {venv_dir} from {base}", flush=True)
    out = subprocess.run([base, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                         capture_output=True, text=True)
    ver = out.stdout.strip()
    if ver and tuple(int(x) for x in ver.split(".")) >= (3, 11):
        print(f"[venv] WARNING: building on Python {ver}. CPA pins torch<2, which "
              f"does not import on 3.11+; expect a dataclass ValueError. Pass "
              f"--venv_python /usr/bin/python3.10 or similar.", flush=True)
    subprocess.run([base, "-m", "venv", venv_dir], check=True)
    py = os.path.join(venv_dir, "bin", "python")
    subprocess.run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
                   check=True)
    print(f"[venv] installing {CPA_PINS} (this pulls torch<2, which is why it "
          f"cannot share the VCPE environment)", flush=True)
    subprocess.run([py, "-m", "pip", "install", "--quiet"] + CPA_PINS, check=True)
    _shim_tkinter(venv_dir, py)
    _shim_jaxlib(py)
    out = subprocess.run(
        [py, "-c", "import torch, cpa, importlib.metadata as m; "
                   "print('torch', torch.__version__); "
                   "print('cpa', m.version('cpa-tools'))"],
        capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip(), flush=True)
    print(f"[venv] ready: {py}", flush=True)


def _shim_tkinter(venv_dir, py):
    """Work around a stray unused import in cpa-tools 0.8.1.

    `cpa/_model.py` line 4 is `from tkinter import N` -- an accidental IDE
    auto-import. `N` is never used anywhere in the package (verified: it is the
    only tkinter reference in cpa, and the `N` occurrences in `_plotting.py` are
    local parameter names), but the import makes CPA unimportable on any Python
    without tkinter, which includes most slim/container images. tkinter is part of
    the standard library but ships as a separate OS package and was not obtainable
    for this interpreter version here.

    Rather than edit the installed package, a minimal stub is placed on the venv's
    path. It provides only the one name CPA imports and nothing else, so if any
    code ever genuinely needs tkinter it fails loudly instead of silently getting
    a fake GUI toolkit.
    """
    import sysconfig
    out = subprocess.run([py, "-c", "import tkinter"], capture_output=True, text=True)
    if out.returncode == 0:
        return                                  # real tkinter present, leave it alone
    sp = subprocess.run(
        [py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True).stdout.strip()
    stub = os.path.join(sp, "tkinter.py")
    with open(stub, "w") as f:
        f.write(
            '"""Minimal stub: cpa-tools 0.8.1 has a stray `from tkinter import N`\n'
            'in cpa/_model.py that it never uses. Only that one name is provided;\n'
            'anything else raises, so a real dependency on tkinter cannot hide here.\n'
            'Written by src/maprna_p3/baseline_cpa.py --prepare_venv.\n"""\n'
            'N = "n"          # the tkinter anchor constant CPA imports and ignores\n\n\n'
            'def __getattr__(name):\n'
            '    # Dunder lookups come from the import machinery itself (__path__,\n'
            '    # __all__, ...) and must fail as a normal AttributeError, or the\n'
            '    # module cannot be imported at all.\n'
            '    if name.startswith("__") and name.endswith("__"):\n'
            '        raise AttributeError(name)\n'
            '    raise ImportError(\n'
            '        f"tkinter.{name} requested, but this is the stub installed to work "\n'
            '        f"around an unused import in cpa-tools. Install real tkinter "\n'
            '        f"(e.g. the python3-tk OS package) if it is genuinely needed.")\n')
    print(f"[venv] wrote a tkinter stub at {stub} (see _shim_tkinter for why)",
          flush=True)


def _shim_jaxlib(py):
    """Re-export `jaxlib.xla_extension.Device` for scvi-tools 0.20.3.

    scvi-tools 0.20.3 does `from jaxlib.xla_extension import Device`. That module
    is gone in current jaxlib, and jaxlib older than 0.4.14 is no longer
    distributed for Python 3.10, so pinning back is not available.

    This is a rename, not a removal: the class is `jaxlib.xla_client.Device`, and
    scvi uses the name only in type annotations (`def to(self, device: Device)`).
    A module that re-exports the real class is therefore faithful -- unlike the
    tkinter stub, nothing here is faked. If the attribute cannot be found, no
    stub is written and the original ImportError stands rather than being masked.
    """
    probe = subprocess.run(
        [py, "-c", "import jaxlib.xla_extension as m; m.Device"],
        capture_output=True, text=True)
    if probe.returncode == 0:
        return                                      # nothing to do
    where = subprocess.run(
        [py, "-c", "import jaxlib.xla_client as c; print(bool(getattr(c, 'Device', None)))"],
        capture_output=True, text=True).stdout.strip()
    if where != "True":
        print("[venv] WARNING: jaxlib.xla_client.Device not found either; leaving "
              "the scvi-tools import error in place rather than faking the symbol.",
              flush=True)
        return
    sp = subprocess.run(
        [py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True).stdout.strip()
    pkg = os.path.join(sp, "jaxlib", "xla_extension")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write(
            '"""Compatibility re-export for scvi-tools 0.20.3.\n\n'
            'scvi-tools 0.20.3 does `from jaxlib.xla_extension import Device`.\n'
            'Current jaxlib exposes the same class as jaxlib.xla_client.Device;\n'
            'this is a rename, and scvi uses the name only in type annotations.\n'
            'The real class is re-exported, so nothing is faked.\n'
            'Written by src/maprna_p3/baseline_cpa.py --prepare_venv.\n"""\n'
            'from jaxlib.xla_client import Device  # noqa: F401\n')
    print(f"[venv] re-exported jaxlib.xla_client.Device as "
          f"jaxlib.xla_extension.Device (see _shim_jaxlib)", flush=True)


# -------------------------------------------------------------- export --------

def stage_export(args):
    """Write everything the CPA run and the later scoring need. VCPE env only."""
    from train_p3 import build_dev_data
    from baseline_common import hvg_symbols, inner_split_items

    if len(args.data_dirs) != 1:
        raise SystemExit("pass exactly one h5ad: CPA is set up per AnnData, and a "
                         "multi-dataset VCPE run has no single CPA counterpart. "
                         "Compare against eval_fair.py's per-dataset rows.")
    data = build_dev_data(args)
    tr, va, te = inner_split_items(data)
    syms = hvg_symbols(data)

    os.makedirs(args.work, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.work, "vcpe.npz"),
        hvg_rows=np.asarray(data["hvg_rows"]),
        dev_te=data["dev_te"], mask_te=data["mask_te"],
        common_fc=data["common_fc"],
        ctrl_feat_rows=np.asarray([di for di, _ in te]),
    )
    train_h5ad = args.train_h5ad or args.data_dirs[0]
    if args.train_h5ad:
        # the comparison is only on the same split if both files hold the same
        # conditions, so this is checked here rather than in the other environment
        import anndata as ad_
        a_ = ad_.read_h5ad(train_h5ad, backed="r")
        have = set(a_.obs["condition"].astype(str)) - {"ctrl"}
        want = {c for _, c in tr + va + te if c.lower() != "ctrl"}
        miss = sorted(want - have)
        del a_
        if miss:
            raise SystemExit(
                f"{train_h5ad} is missing {len(miss)} of the {len(want)} conditions "
                f"the split covers, e.g. {miss[:5]}. Both files must come from the "
                "same source and the same --min_cells, or the split is not shared.")
    meta = dict(
        h5ad=os.path.abspath(args.data_dirs[0]),
        train_h5ad=os.path.abspath(train_h5ad),
        use_mask=True,  # scored through the intersected measured mask, unconditionally
        # One canonical protocol statement, carried across the environment
        # boundary. The score stage runs with its own argv (--work, --out_json),
        # so its provenance cannot describe the experiment; without this a reader
        # -- and the assembler -- could not tell which panel or split produced it.
        protocol=dict(data_dirs=[os.path.abspath(d) for d in args.data_dirs],
                      n_hvg=args.n_hvg, hvg_from=getattr(args, "hvg_from", "train"),
                      split_by=args.split_by, test_frac=args.test_frac,
                      min_cells=args.min_cells, seed=args.seed, use_mask=True),
        train_h5ad_is_cell_level=bool(args.train_h5ad),
        split_by=args.split_by, seed=args.seed, n_hvg=args.n_hvg,
        hvg_symbols=syms,
        train_conditions=sorted({c for _, c in tr if c.lower() != "ctrl"}),
        valid_conditions=sorted({c for _, c in va if c.lower() != "ctrl"}),
        test_conditions=[c for _, c in te],
        test_dataset_idx=[int(di) for di, _ in te],
    )
    with open(os.path.join(args.work, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] {args.work}: train {len(meta['train_conditions'])} / "
          f"valid {len(meta['valid_conditions'])} / "
          f"test {len(meta['test_conditions'])} conditions | "
          f"panel {len(syms)} genes", flush=True)
    print("[export] next: run the `run` stage with the CPA venv's python", flush=True)


# ----------------------------------------------------------------- run --------

def stage_run(args):
    """Train CPA and write predicted expression. Runs in the CPA venv."""
    import anndata as ad
    import cpa
    import pandas as pd

    with open(os.path.join(args.work, "meta.json")) as f:
        meta = json.load(f)
    adata = ad.read_h5ad(meta.get("train_h5ad", meta["h5ad"]))
    if "gene_name" in adata.var.columns:
        adata.var_names = adata.var["gene_name"].astype(str)
        adata.var_names_make_unique()
    # CPA is a latent-variable model over cells; one pseudobulk row per condition
    # leaves it nothing to model within a condition. Say so here rather than let it
    # train on 1-sample groups and quietly report a number.
    npc = adata.obs["condition"].astype(str).value_counts()
    if (npc < 2).any() and not meta.get("train_h5ad_is_cell_level"):
        print(f"[cpa] WARNING: {int((npc < 2).sum())} of {len(npc)} conditions have "
              "a single row. CPA is a cell-level model and this is what a pseudobulk "
              "export looks like to it; build a cell file with ingest_scperturb "
              "--mode cells and re-export with --train_h5ad.", flush=True)

    # CPA reads the split from an obs column, so VCPE's split goes in verbatim
    # rather than being approximated by one of CPA's own splitters.
    # GEARS-format data labels a single-gene perturbation "GENE+ctrl". CPA parses
    # its perturbation key on "+" and counts every part as a perturbation, so
    # "MYC+ctrl" reads as a two-way combination of MYC with the control -- which
    # produces ragged combination lengths against plain "ctrl" rows and fails in
    # np.vstack. Strip the suffix so CPA sees what the label actually means: one
    # perturbed gene. The split sets are translated the same way, so the two
    # cannot disagree.
    def to_cpa_label(c):
        c = str(c)
        if c.lower() == "ctrl":
            return "ctrl"
        parts = [x for x in c.split("+") if x.lower() != "ctrl"]
        return "+".join(parts) if parts else "ctrl"

    tr = {to_cpa_label(c) for c in meta["train_conditions"]}
    va = {to_cpa_label(c) for c in meta["valid_conditions"]}
    te = {to_cpa_label(c) for c in meta["test_conditions"]}
    raw_cond = adata.obs["condition"].astype(str)
    adata.obs["condition"] = raw_cond.map(to_cpa_label)
    cond = adata.obs["condition"].astype(str)
    is_ctrl = cond.str.lower() == "ctrl"
    max_comb = max(1, max(len(c.split("+")) for c in set(cond) if c != "ctrl"))
    print(f"[cpa] perturbation labels: {len(set(cond)) - 1} distinct + ctrl | "
          f"max combination length {max_comb}", flush=True)

    split = pd.Series("unused", index=adata.obs.index, dtype=object)
    split[cond.isin(tr)] = "train"
    split[cond.isin(va)] = "valid"
    split[cond.isin(te)] = "ood"
    # Control cells belong to training: they define the baseline CPA decodes
    # against, and holding them out would change the task rather than the model.
    split[is_ctrl] = "train"
    adata.obs["split"] = split.values
    counts = adata.obs["split"].value_counts().to_dict()
    print(f"[cpa] split cells: {counts}", flush=True)
    if counts.get("ood", 0) == 0:
        raise SystemExit("no held-out cells: the exported test conditions do not "
                         "appear in this h5ad's `condition` column.")

    adata.obs["dose"] = np.where(is_ctrl, 0.0, 1.0)   # CPA expects a dose axis
    if "cell_type" not in adata.obs.columns:
        adata.obs["cell_type"] = "cells"

    # cpa-tools 0.8.1 builds a perturbation -> SMILES map unconditionally in
    # setup_anndata (the block runs whenever the class attribute is None,
    # regardless of whether smiles_key was passed), so gene perturbations hit
    # `KeyError: None`. Pre-setting the map to empty skips that block.
    #
    # This is the honest workaround rather than inventing a SMILES column:
    # _model.py:111 shows the map is consumed ONLY under
    # `use_rdkit_embeddings`, which is left off, so CPA uses its learned
    # per-perturbation embeddings -- the correct representation for gene
    # knockdowns. Supplying a placeholder SMILES string instead would make every
    # perturbation chemically identical if that path were ever enabled.
    cpa.CPA.pert_smiles_map = {}

    cpa.CPA.setup_anndata(
        adata, perturbation_key="condition", control_group="ctrl",
        dosage_key="dose", categorical_covariate_keys=["cell_type"],
        is_count_data=False, max_comb_len=max_comb)
    model = cpa.CPA(adata=adata, split_key="split", train_split="train",
                    valid_split="valid", test_split="ood",
                    n_latent=args.n_latent)
    print(f"[cpa] training up to {args.max_epochs} epochs", flush=True)
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size,
                early_stopping_patience=args.patience, check_val_every_n_epoch=5,
                save_path=os.path.join(args.work, "cpa_model"))

    model.predict(adata, batch_size=args.batch_size)
    key = next((k for k in ("CPA_pred", "CPA_pred_expression") if k in adata.obsm),
               None)
    if key is None:
        raise SystemExit(f"CPA wrote no prediction to obsm; keys: {list(adata.obsm)}")
    pred = np.asarray(adata.obsm[key])

    # One predicted profile per held-out condition: the mean over that
    # condition's cells, which is the level VCPE's targets are defined at.
    out = np.zeros((len(meta["test_conditions"]), adata.n_vars), dtype=np.float32)
    n_cells = []
    for i, c in enumerate(meta["test_conditions"]):
        sel = (cond == to_cpa_label(c)).values
        n_cells.append(int(sel.sum()))
        if sel.any():
            out[i] = pred[sel].mean(axis=0)

    ctrl_mean = np.asarray(adata[is_ctrl.values].X.mean(axis=0)).ravel()
    np.savez_compressed(os.path.join(args.work, "cpa_pred.npz"),
                        pred_expr=out, ctrl_mean=ctrl_mean,
                        gene_names=np.array(list(adata.var_names), dtype=object),
                        n_cells_per_condition=np.asarray(n_cells))
    print(f"[cpa] wrote predictions for {len(out)} conditions "
          f"(cells per condition: min {min(n_cells)}, max {max(n_cells)})",
          flush=True)
    print("[cpa] next: run the `score` stage with the VCPE environment's python",
          flush=True)


# --------------------------------------------------------------- score --------

def stage_score(args):
    """Convert CPA's expression to VCPE's residual and score it. VCPE env only."""
    from baseline_common import (align_to_panel, expression_to_dev,
                                 fairness_block, intersect_masks, score)
    from provenance import write_json

    with open(os.path.join(args.work, "meta.json")) as f:
        meta = json.load(f)
    v = np.load(os.path.join(args.work, "vcpe.npz"), allow_pickle=True)
    p = np.load(os.path.join(args.work, "cpa_pred.npz"), allow_pickle=True)

    syms = meta["hvg_symbols"]
    gene_names = list(p["gene_names"])
    common_fc = v["common_fc"]
    ctrl_panel, ctrl_found = align_to_panel(p["ctrl_mean"], gene_names, syms)

    pred_dev = np.zeros_like(v["dev_te"])
    coverage = np.zeros_like(v["mask_te"], dtype=bool)
    for i in range(len(pred_dev)):
        expr_panel, found = align_to_panel(p["pred_expr"][i], gene_names, syms)
        pred_dev[i] = expression_to_dev(expr_panel, ctrl_panel, common_fc)
        coverage[i] = found & ctrl_found

    mask = intersect_masks(v["mask_te"], coverage)
    rep = dict(
        model="CPA",
        n_test_conditions=int(len(pred_dev)),
        n_cells_per_condition=[int(x) for x in p["n_cells_per_condition"]],
        vcpe_measured_fraction=float(np.mean(v["mask_te"])),
        cpa_coverage_fraction=float(np.mean(coverage)),
        intersected_mask_fraction=float(np.mean(mask)),
        cpa=score(v["dev_te"], pred_dev, mask, seed=meta["seed"]),
        protocol=meta.get("protocol", {}),
        config=dict(split_by=meta["split_by"], seed=meta["seed"],
                    n_hvg=meta["n_hvg"]),
        environment_note=(
            "CPA was run in a separate virtual environment because cpa-tools pins "
            "torch<2.0.0 while this repository requires torch>=2.1. Installing it "
            "alongside VCPE downgrades torch and breaks the model code. Record "
            "both environments next to this number."),
        fairness=fairness_block(extra=([
            "Input granularity differs on purpose: CPA trained on individual cells "
            "while the P3 head trains on the pseudobulk mean of the same cells. The "
            "split, panel, predicted quantity and metric are identical. CPA models "
            "variation within a condition, so a pseudobulk input would not be a "
            "fairer comparison, only a meaningless one.",
        ] if meta.get("train_h5ad_is_cell_level") else [])),
    )
    c = rep["cpa"]
    print("\n=== CPA on VCPE's split, panel, target and metrics ===", flush=True)
    print(f"  pearson_dev        {c['pearson_dev']:.4f}  "
          f"CI95 [{c['pearson_dev_ci95'][0]:.4f}, {c['pearson_dev_ci95'][1]:.4f}]",
          flush=True)
    print(f"  pearson_dev_pooled {c['pearson_dev_pooled']:.4f}", flush=True)
    print(f"  top50_dev          {c['top50_dev']}", flush=True)
    print(f"  mask fraction      {rep['intersected_mask_fraction']:.3f} "
          f"(VCPE measured {rep['vcpe_measured_fraction']:.3f} x CPA coverage "
          f"{rep['cpa_coverage_fraction']:.3f})", flush=True)
    print("\nCompare against the VCPE head from the SAME --split_by / --seed / "
          "--n_hvg, and read `fairness.not_controlled` before quoting either.",
          flush=True)
    if args.out_json:
        write_json(args.out_json, rep, args=args)
        print(f"wrote {args.out_json}", flush=True)


# ---------------------------------------------------------------- cli ---------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", nargs="?", choices=["export", "run", "score"])
    p.add_argument("--work", default="cpa_run",
                   help="directory used as the interface between the stages")
    p.add_argument("--prepare_venv", action="store_true")
    p.add_argument("--venv", default=".venv_cpa")
    p.add_argument("--venv_python", default=None,
                   help="interpreter to build the CPA venv from; defaults to "
                        "python3.10, which CPA's torch<2 pin requires")
    # export
    p.add_argument("--data_dirs", nargs="+", default=[])
    p.add_argument("--esm_table", default="")
    p.add_argument("--train_h5ad", default=None,
                   help="cell-level h5ad CPA trains on. Default: the same file as "
                        "--data_dirs. CPA is a latent-variable model over cells, so "
                        "one pseudobulk row per condition leaves it nothing to model "
                        "within a condition. The split, the gene panel and the scored "
                        "quantity still come from --data_dirs, the file VCPE reads; "
                        "only the training input differs, and the fairness block says "
                        "so.")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--hvg_from", choices=("train", "all"), default="train",
                   help="which perturbations the response panel is ranked on. Must "
                        "match the run being compared against: the panel is part of "
                        "the protocol, not a detail (docs/ERRATA.md E15).")
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--inner_val_frac", type=float, default=0.15)
    p.add_argument("--split_by", default="target_gene",
                   choices=["target_gene", "pert"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--knn_k", type=int, default=10)
    # run
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_latent", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    # score
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    if args.prepare_venv:
        prepare_venv(args.venv, args.venv_python)
        if not args.stage:
            return
    if not args.stage:
        raise SystemExit("pass a stage: export | run | score (see the module "
                         "docstring for the order and which environment each "
                         "one needs)")
    {"export": stage_export, "run": stage_run, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    main()
