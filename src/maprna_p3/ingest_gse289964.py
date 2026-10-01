#!/usr/bin/env python
"""GSE289964 ingestion: in vivo mouse liver Scarb1 gapmer ASO (3 modifications
× 3 timepoints, 48 samples).

Data: local AmpliSeq bcmatrix (48 samples × 23,930 mouse genes, counts) + the
GEO series_matrix design table (fetched online, already verified):
  PBS 12 (control) | LX-A1656 / LX-A2003 / LX-A3928 (all target Scarb1) ×
  24h/72h/168h × 4 reps.

Convention:
  condition = "SCARB1" for every ASO sample -- a single target, so that
      load_kd_datasets sees one perturbation and pools the chemistries. The
      detailed label "SCARB1@{modification}{timepoint}" goes in
      perturbation_raw, where 3 modifications x 3 timepoints give 9 distinct
      combinations (48 samples = 12 x 4 replicates, PBS included).
      (Up to v4 this line claimed condition itself carried the detailed label
      over "12 perturbation conditions"; neither matched the code.)
  control = 1 for PBS only (12 samples)
  X = CP10K + log1p (counts normalized directly)
  genes: mouse symbol -> human ortholog (MGI capitalize-first-letter
        convention), intersected with the ESM table; genes that cannot be
        mapped are dropped and the mapping rate is reported.

Sanity: Scarb1 should be markedly down-regulated in each ASO group vs PBS (the
gapmer knocks it down directly).

Usage: python ingest_gse289964.py [--list-only]
"""
import argparse
import gzip
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data", "aso_tx_validation"))
BCM_FP = os.path.join(DATA, "GSE289964_AmpliSeq_bcmatrix.tsv.gz")
ESM_FP = os.path.abspath(os.path.join(HERE, "..", "..", "data",
                                      "drive_weights",
                                      "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"))
OUT_FP = os.path.join(DATA, "gse289964_proc.h5ad")
SERIES_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/"
              "GSE289964/matrix/GSE289964_series_matrix.txt.gz")


def fetch_design():
    with urllib.request.urlopen(SERIES_URL, timeout=60) as r:
        data = gzip.decompress(r.read()).decode("utf-8", errors="replace")
    titles = None
    for line in data.split("\n"):
        if line.startswith("!Sample_title"):
            titles = [p.strip('"') for p in line.split("\t")[1:]]
            break
    assert titles and len(titles) == 48, f"bad sample count: {len(titles) if titles else 0}"
    samples = []
    for t in titles:
        tl = t.lower()
        mod = next((m for m in ("a1656", "a2003", "a3928") if m in tl), None)
        day = next((d for d in ("24h", "72h", "168h") if d in tl), None)
        if "pbs" in tl:
            samples.append(dict(title=t, mod="PBS", day=day or ""))
        else:
            assert mod and day, f"cannot parse sample: {t}"
            samples.append(dict(title=t, mod=mod, day=day))
    return samples


def mouse_to_human(syms, esm_syms):
    """MGI naming convention: the mouse symbol is capitalized first-letter only;
    upper-casing all of it gives the human ortholog (1:1 in the vast majority).
    """
    mapping, unmapped = {}, 0
    for s in syms:
        h = s.upper()          # Scarb1 -> SCARB1, Mup3 -> MUP3
        if h in esm_syms:
            mapping[s] = h
        else:
            unmapped += 1
    return mapping, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    samples = fetch_design()
    from collections import Counter
    cnt = Counter(f"{s['mod']}@{s['day']}" if s["mod"] != "PBS" else "PBS"
                  for s in samples)
    print("[design]", dict(cnt), flush=True)
    if args.list_only:
        return

    # ---- matrix ----
    df = pd.read_csv(BCM_FP, sep="\t", index_col=0)
    df.columns = [c.strip() for c in df.columns]
    print(f"[data] {df.shape} | gene head: {list(df.index[:3])}", flush=True)

    # ---- ESM symbol set + mouse->human mapping ----
    import torch
    tab = torch.load(ESM_FP, map_location="cpu", weights_only=False)
    esm_syms = set(tab.keys())
    mapping, unmapped = mouse_to_human(list(df.index), esm_syms)
    print(f"[map] mouse genes {len(df.index)} -> human ortholog hits "
          f"{len(mapping)} (unmapped {unmapped})", flush=True)

    # ---- aggregate to human symbol (sum over same-named multiple
    # transcripts / multiple mouse genes) ----
    genes = sorted(set(mapping.values()))
    gi = {g: i for i, g in enumerate(genes)}
    R = np.zeros((df.shape[1], len(genes)), dtype=np.float32)
    col_of_gene = {}
    for mouse_sym, human_sym in mapping.items():
        col_of_gene.setdefault(human_sym, []).append(mouse_sym)
    sub = df.loc[[m for m in mapping]]
    for h_sym, m_syms in col_of_gene.items():
        R[:, gi[h_sym]] = sub.loc[m_syms].sum(axis=0).values
    # CP10K + log1p
    tot = R.sum(axis=1, keepdims=True) + 1e-6
    X = np.log1p(R / tot * 1e4)
    print(f"[agg] human genes: {len(genes)} | samples: {R.shape[0]}", flush=True)

    # ---- obs ----
    obs_rows, order = [], []
    cond_list, ctrl_list = [], []
    for j, s in enumerate(samples):
        sid = s["title"].replace(" ", "_")
        if s["mod"] == "PBS":
            cond, ctrl, praw = "ctrl", 1, "ctrl"
        else:
            # resolve_pert_row looks the whole condition string up in the ESM
            # symbol table (human, upper-case), so the modification/timepoint
            # information is kept in perturbation_raw; the 9 conditions are
            # merged into one and the same target (the model has no
            # modification axis, so this is semantically correct and on the
            # same convention as GSE183535 condition=MYC)
            cond, ctrl, praw = "SCARB1", 0, \
                f"SCARB1@{s['mod']}{s['day']}"
        cond_list.append(cond); ctrl_list.append(ctrl)
        obs_rows.append(dict(
            sample_id=sid, condition=cond, control=ctrl, perturbation_raw=praw,
            cell_line="MouseLiver_invivo", perturbation_type="ASO",
            tissue_type="liver_invivo", mod=s["mod"], timepoint=s["day"]))
        order.append(sid)
    obs = pd.DataFrame(obs_rows).set_index("sample_id").loc[order]

    import anndata as ad
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.write_h5ad(OUT_FP)
    print(f"[done] {OUT_FP} {adata.shape} | perturbation conditions "
          f"{len(set(c for c in cond_list if c != 'ctrl'))} | ctrl "
          f"{sum(1 for c in ctrl_list if c == 1)}", flush=True)

    # ---- sanity: Scarb1 knockdown direction (each modification 24h vs PBS) ----
    import torch as _t
    sym2row_sanity = None
    if "SCARB1" in gi:
        i = gi["SCARB1"]
        pbs_m = X[[j for j, s in enumerate(samples) if s["mod"] == "PBS"], i].mean()
        for mod in ("a1656", "a2003", "a3928"):
            m = X[[j for j, s in enumerate(samples)
                   if s["mod"] == mod and s["day"] == "24h"], i].mean()
            fc = np.log2((np.expm1(m) + 1e-6) / (np.expm1(pbs_m) + 1e-6))
            print(f"[sanity] SCARB1 {mod} 24h vs PBS: log2FC {fc:+.2f} "
                  f"(the gapmer should be markedly negative)",
                  flush=True)
    else:
        print("[sanity] SCARB1 is not in the mapping result -- check the "
              "mapping!", flush=True)


if __name__ == "__main__":
    main()
