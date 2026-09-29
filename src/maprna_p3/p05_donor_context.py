#!/usr/bin/env python
"""P0.5 primary-liver context fix: donor ctrl feature replacement comparison
experiment (runs on CPU).

The virtual-liver G0 ruling found that the hepg2 cancer-line donor context
brings the organ-level direction-consistency rate down to only 44% (<60%).
Under an entirely identical model / panel / protocol, this script replaces the
donor ctrl features only (the cf vector + the pe scalar) and compares how three
donors perform against the GSE289964 organ measurement (in vivo whole liver):

  hepg2    -- baseline (the current G0 convention, z-scored within the hepg2
              dataset)
  invivo   -- the mean of the GSE289964 PBS control group (log1pTPM, the
              baseline of the same experiment)
  gtex     -- GTEx v10 liver tissue median TPM (log1p, a public primary-tissue
              reference)

Usage (CPU):
  python p05_donor_context.py \
    --ckpt ../../p3_v21e/ckpt_p3_best_dev.pt \
    --hepg2_h5ad ../../data/scperturb_proc/NadigOConner2024_hepg2_proc.h5ad \
    --invivo_h5ad ../../data/aso_tx_validation/gse289964_proc.h5ad \
    --gtex_gct ../../data/gtex_v10_gene_median_tpm.gct.gz \
    --esm_table ../../data/drive_weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
    --string_neighbors ../../data/l2/string_neighbors.npy \
    --rna_encoder_ckpt ../../p3_v21e/rna_enc_from_ckpt.pt \
    --fasta ../../data/gene_transcripts.fa \
    --targets SCARB1 APOC3 ALB
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
for _sib in ("maprna_p1", "maprna_p2", "maprna_p3"):
    _p = os.path.abspath(os.path.join(HERE, "..", _sib))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from ds_knockdown import load_kd_datasets, build_symbol2row, resolve_pert_row  # noqa: E402
from model_dev import DeviationModel, load_esm_matrix, load_dev_state  # noqa: E402


def build_model(args, device):
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_ds = int(ck["model_state_dict"]["ds_emb.weight"].shape[0])
    hvg_rows = np.asarray(ck["hvg_rows"], dtype=np.int64)
    common_fc = np.asarray(ck["common_fc"], dtype=np.float32)
    sym2row, _ = build_symbol2row(args.esm_table)
    sym_by_row = {int(r): s for s, r in sym2row.items()}
    esm = load_esm_matrix(args.esm_table)
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "maprna_p2")))
    from rna_encoder import RNAEncoder
    rna_enc = RNAEncoder(d_model=256, max_len=600).to(device).eval()
    state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "rna_encoder" in state:
        state = state["rna_encoder"]
    rna_enc.load_state_dict(state)
    model = DeviationModel(esm, hvg_rows, n_ds=n_ds, rna_encoder=rna_enc).to(device).eval()
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
    load_dev_state(model, ck)
    syms = [sym_by_row.get(int(r)) for r in hvg_rows]
    return ck, n_ds, hvg_rows, common_fc, sym2row, sym_by_row, model, syms


def donor_hepg2(args, sym2row, sym_by_row, hvg_rows, targets):
    """Baseline: ctrl from inside the hepg2 dataset (log1p CP10K)."""
    kd = load_kd_datasets([args.hepg2_h5ad], sym2row, min_cells=args.min_cells)
    r2c = {int(r): c for c, r in enumerate(kd["row_of_gene"][0])}
    cm = np.asarray(kd["X_ctrl"][0].mean(axis=0), dtype=np.float32)
    sym2val = {}
    for i, r in enumerate([int(x) for x in kd["row_of_gene"][0]]):
        s = sym_by_row.get(r)
        if s is not None:
            sym2val[s] = float(cm[i])
    return sym2val


def donor_invivo(args, hvg_rows):
    """Mean of the GSE289964 PBS controls (log1pTPM). Caveat: the donor ctrl
    contains no perturbation-response information at all, but it comes from the
    same experiment as this exam -- the results are annotated with that caveat,
    and the official version can use leave-one-modification-out isolation to
    validate.
    """
    import anndata as ad
    a = ad.read_h5ad(args.invivo_h5ad)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    ctrl = np.where(a.obs["control"].values == 1)[0]
    cm = X[ctrl].mean(axis=0)
    sym2val = {str(g): float(v) for g, v in zip(map(str, a.var_names), cm)}
    print(f"  [donor invivo] ctrl samples {len(ctrl)} | genes {len(sym2val)}",
          flush=True)
    return sym2val


def donor_gtex(args, hvg_rows):
    """GTEx v10 liver median TPM (log1p). gct: Name / Description(symbol) / 54 tissue cols."""
    tissue_col, rows = None, {}
    with gzip.open(args.gtex_gct, "rt") as f:
        f.readline(); f.readline()  # version + shape lines
        header = f.readline().rstrip("\n").split("\t")
        tissue_col = header.index("Liver")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sym = parts[1].split(".")[0]
            try:
                rows[sym] = float(parts[tissue_col])
            except ValueError:
                pass
    sym2val = {g: float(np.log1p(v)) for g, v in rows.items()}
    print(f"  [donor gtex] liver median TPM genes {len(sym2val)}", flush=True)
    return sym2val


def eval_g0(pred_by_sym, tag):
    """The two G0 measures: in-panel DEG direction-consistency rate + spearman
    (same protocol as the P0 report).
    """
    from scipy.stats import spearmanr
    import anndata as ad
    a = ad.read_h5ad(os.path.join(HERE, "..", "..", "data", "aso_tx_validation",
                                  "gse289964_proc.h5ad"))
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    genes = list(map(str, a.var_names)); gidx = {g: i for i, g in enumerate(genes)}
    obs = a.obs
    ctrl = np.where(obs["control"] == 1)[0]
    tp_c = np.expm1(X[ctrl]).mean(0)
    ag, rg, cov = [], [], []
    for mod in ["a1656", "a2003", "a3928"]:
        for tp in sorted(set(obs["timepoint"])):
            pi = np.where((obs["mod"] == mod) & (obs["timepoint"] == tp))[0]
            tp_p = np.expm1(X[pi]).mean(0)
            lfc = np.log2((tp_p + 1.0) / (tp_c + 1.0))
            deg = [g for g in genes if tp_c[gidx[g]] >= 2.0 and abs(lfc[gidx[g]]) >= 1.0]
            in_panel = [g for g in deg if g in pred_by_sym]
            if len(in_panel) < 10:
                continue
            agree = float(np.mean([np.sign(pred_by_sym[g]) == np.sign(lfc[gidx[g]])
                                   for g in in_panel]))
            pv = np.array([pred_by_sym[g] for g in in_panel])
            mv = np.array([lfc[gidx[g]] for g in in_panel])
            rho = float(spearmanr(pv, mv).statistic)
            ag.append(agree); rg.append(rho)
            cov.append(len(in_panel) / max(1, len(deg)))
            print(f"    {mod+'@'+tp:12s} nDEG {len(deg):>4d} | ∩ {len(in_panel):>3d} | "
                  f"consistency {agree:.0%} | rho {rho:+.3f}", flush=True)
    print(f"  ==> [{tag}] mean consistency rate {np.mean(ag):.0%} | "
          f"spearman {np.mean(rg):+.3f} | "
          f"coverage {np.mean(cov):.1%}", flush=True)
    return float(np.mean(ag)), float(np.mean(rg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hepg2_h5ad", required=True)
    ap.add_argument("--invivo_h5ad", required=True)
    ap.add_argument("--gtex_gct", default="")
    ap.add_argument("--esm_table", required=True)
    ap.add_argument("--string_neighbors", default="")
    ap.add_argument("--rna_encoder_ckpt", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--donor_ds_idx", type=int, default=5)
    ap.add_argument("--targets", nargs="+", default=["SCARB1", "APOC3", "ALB"])
    ap.add_argument("--min_cells", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    ck, n_ds, hvg_rows, common_fc, sym2row, sym_by_row, model, syms = build_model(args, device)
    print(f"ckpt: epoch {ck.get('epoch')} | n_ds={n_ds} | panel {len(hvg_rows)}", flush=True)

    symbol_map, ensembl_map = None, None

    for tgt in args.targets:
        row = resolve_pert_row(sym2row, tgt)
        if row is None or int(row) < 0:
            print(f"[{tgt}] cannot be resolved, skipping", flush=True)
            continue
        print(f"\n===== target {tgt} =====", flush=True)
        for tag, builder in [("hepg2", lambda: donor_hepg2(args, sym2row, sym_by_row, hvg_rows, [tgt])),
                             ("invivo", lambda: donor_invivo(args, hvg_rows)),
                             ("gtex", lambda: donor_gtex(args, hvg_rows))]:
            if tag == "gtex" and not (args.gtex_gct and os.path.exists(args.gtex_gct)):
                continue
            sym2val = builder()
            ctrl_vec = np.array([sym2val.get(s, 0.0) for s in syms], dtype=np.float32)
            cf = ((ctrl_vec - ctrl_vec.mean()) / (ctrl_vec.std() + 1e-6))[None, :]
            pe = float(sym2val.get(tgt, 0.0))
            if symbol_map is None:
                from rna_encoder import encode_seq, load_fasta_symbol_seqs, lookup_seq
                symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
            seq = lookup_seq(tgt, symbol_map, ensembl_map)
            if seq:
                ids, m = encode_seq(seq, 600)
            else:
                ids, m = [6] + [0] * 599, [1] + [0] * 599
            with torch.no_grad():
                pr = torch.tensor([int(row)], dtype=torch.long).to(device)
                dsi = torch.tensor([args.donor_ds_idx], dtype=torch.long).to(device)
                rna_e = model.rna_encoder(torch.tensor([ids], device=device),
                                          torch.tensor([m], device=device).bool())[0].float().unsqueeze(0)
                dev = model(pr, dsi, torch.from_numpy(cf).float().to(device),
                            pert_ctrl_expr=torch.tensor([pe], dtype=torch.float32).to(device),
                            rna_emb=rna_e)
            pred_fc = dev.cpu().numpy()[0] + common_fc
            pred_by_sym = {s: float(v) for s, v in zip(syms, pred_fc) if s}
            print(f"  [{tag}] pe={pe:.3f} | fc std {pred_fc.std():.3f} max|.| {np.abs(pred_fc).max():.3f}",
                  flush=True)
            eval_g0(pred_by_sym, f"{tgt}/{tag}")


if __name__ == "__main__":
    main()
