#!/usr/bin/env python
"""G0 prerequisite task: export the full 2000-HVG prediction vectors of the
liver-lineage anchor targets (prerequisite for the virtual liver P1).

Rebuilds the engine from the p3_v21e ckpt (same protocol as eval_aso_domain:
donor=hepg2 ctrl z-score, the rna_encoder fine-tuned inside the ckpt, scalar
gating pe, STRING adjacency) and, for each requested target, writes the full
pred_fc = dev + common_fc (2000 dims, symbols corresponding to hvg_rows).

Note on the ds_emb row index: for n_ds=9 the hepg2 row index = 5 (its index in
data_dirs at training time); even though only the single hepg2 dataset is
loaded here, and only for the donor ctrl, dsi must still be passed as 5.

Usage (GPU):
  python export_full_vectors.py \
    --ckpt $BASE/p3_v21e/ckpt_p3_best_dev.pt \
    --hepg2_h5ad $BASE/data/scperturb/NadigOConner2024_hepg2_proc.h5ad \
    --esm_table $BASE/weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
    --string_neighbors $BASE/data/l2/string_neighbors.npy \
    --rna_encoder_ckpt $BASE/p2_align/rna_encoder_aligned.pt \
    --fasta $BASE/data/gene_transcripts.fa \
    --donor_ds_idx 5 \
    --out $BASE/full_vectors_v21e.json
The product is transferred back to the local machine for the G0 ruling (re-test
of DEG coverage / direction-consistency rate).
"""
import argparse
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

TARGETS = ["SCARB1", "APOC3", "ALB", "MYC", "ACTN1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--hepg2_h5ad", required=True)
    ap.add_argument("--esm_table", required=True)
    ap.add_argument("--string_neighbors", default="")
    ap.add_argument("--rna_encoder_ckpt", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--donor_ds_idx", type=int, default=5,
                    help="index of hepg2 in the training data_dirs (the "
                         "ds_emb row index), p3_v21e=5")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--min_cells", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    n_ds = int(ck["model_state_dict"]["ds_emb.weight"].shape[0])
    epoch = ck.get("epoch")
    hvg_rows = np.asarray(ck["hvg_rows"], dtype=np.int64)
    common_fc = np.asarray(ck["common_fc"], dtype=np.float32)
    print(f"ckpt: epoch {epoch} | n_ds={n_ds} | hvg panel {len(hvg_rows)}", flush=True)

    sym2row, _ = build_symbol2row(args.esm_table)
    sym_by_row = {int(r): s for s, r in sym2row.items()}
    esm = load_esm_matrix(args.esm_table)

    # donor(hepg2) ctrl over ckpt hvg panel (within-dataset z-score, same as
    # train_p3)
    kd = load_kd_datasets([args.hepg2_h5ad], sym2row, min_cells=args.min_cells)
    di = 0  # only the single hepg2 dataset was loaded
    r2c = {int(r): c for c, r in enumerate(kd["row_of_gene"][di])}
    cm = np.asarray(kd["X_ctrl"][di].mean(axis=0), dtype=np.float32)
    ctrl_vec = np.array([cm[r2c.get(hr, -1)] if r2c.get(hr, -1) >= 0 else 0.0
                         for hr in hvg_rows], dtype=np.float32)
    cf = ((ctrl_vec - ctrl_vec.mean()) / (ctrl_vec.std() + 1e-6))[None, :]

    # rna_encoder: the fine-tuned weights inside the ckpt overwrite the
    # Stage-A initialization
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "maprna_p2")))
    from rna_encoder import RNAEncoder, encode_seq, load_fasta_symbol_seqs, lookup_seq
    rna_enc = RNAEncoder(d_model=256, max_len=600).to(device).eval()
    state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "rna_encoder" in state:
        state = state["rna_encoder"]
    rna_enc.load_state_dict(state)

    model = DeviationModel(esm, hvg_rows, n_ds=n_ds, rna_encoder=rna_enc).to(device).eval()
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
    load_dev_state(model, ck)

    # hvg panel symbol table
    syms = [sym_by_row.get(int(r)) for r in hvg_rows]
    hepg2_sym2col = {}
    for i, r in enumerate([int(x) for x in kd["row_of_gene"][di]]):
        sname = sym_by_row.get(r)
        if sname is not None:
            hepg2_sym2col[sname] = i

    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    out = {"meta": {"ckpt_epoch": epoch, "n_ds": n_ds, "panel": len(hvg_rows),
                    "donor_ds_idx": args.donor_ds_idx,
                    "symbols": syms, "common_fc": common_fc.tolist()}}
    for tgt in args.targets:
        row = resolve_pert_row(sym2row, tgt)
        if row is None or int(row) < 0:
            print(f"[{tgt}] target gene cannot be resolved, skipping", flush=True)
            continue
        seq = lookup_seq(tgt, symbol_map, ensembl_map)
        if seq:
            ids, m = encode_seq(seq, 600)
        else:
            ids, m = [6] + [0] * 599, [1] + [0] * 599
        col = hepg2_sym2col.get(sym_by_row.get(int(row)))
        pe = float(cm[col]) if col is not None else 0.0
        with torch.no_grad():
            pr = torch.tensor([int(row)], dtype=torch.long).to(device)
            dsi = torch.tensor([args.donor_ds_idx], dtype=torch.long).to(device)
            rna_e = model.rna_encoder(torch.tensor([ids], device=device),
                                      torch.tensor([m], device=device).bool())[0].float().unsqueeze(0)
            dev = model(pr, dsi, torch.from_numpy(cf).float().to(device),
                        pert_ctrl_expr=torch.tensor([pe], dtype=torch.float32).to(device),
                        rna_emb=rna_e)
        pred_fc = dev.cpu().numpy()[0] + common_fc
        out[tgt] = {"pe": pe, "rna_emb": "seq" if seq else "CLS-fallback",
                    "fc": pred_fc.tolist()}
        in_panel = sym_by_row.get(int(row)) in syms
        self_v = pred_fc[syms.index(sym_by_row.get(int(row)))] if in_panel else None
        print(f"[{tgt}] pe={pe:.3f} | self={('%.3f' % self_v) if self_v is not None else 'off-panel'} | "
              f"fc std={pred_fc.std():.3f} max|.|={np.abs(pred_fc).max():.3f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"DONE -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
