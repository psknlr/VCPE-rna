#!/usr/bin/env python
"""VCPE V2-0a: expression gating of the target gene's self-response in the
platform cache (post-process, no GPU re-inference required).

Background (2026-09-04 diagnosis): the is_target feature of VCPE P2.3-B learned
a "the perturbed gene itself is strongly down-regulated" prior from the CRISPRi
training data, which becomes distorted when transferred to a low-expression
context (e.g. APOC3 has near-zero expression in K562 and self_fc=-0.60 is still
predicted; ALB/TTR/SERPINA1 are distorted in the same way). In CRISPRi the guide
directly represses the promoter and the target gene is necessarily silenced;
this is a genuine difference between modalities -- RNA that is not there cannot
be knocked down.

Gating rules (they act only on the target gene's own entry, the self-entry, and
leave trans responses untouched):
  1) panel genes (measured in the context h5ad ctrl): gate = min(1,
     expr_log1p / FLOOR); expr is the pseudobulk mean over ctrl cells (in
     CP10K+log1p space), FLOOR defaults to 0.7.
  2) off-panel genes (not contained in the Perturb-seq panel, which is exactly
     where tissue-specific genes cluster): use the local Protein Atlas tissue
     specificity -- if the category is "Tissue enriched" and the lineage of the
     enriched tissue does not match the queried context's lineage (e.g. a
     liver-enriched gene × the K562 blood lineage), gate = 0.15; everything
     else (Group enriched / Tissue enhanced / low specificity) is not gated.
  Finally self_fc *= gate, and top_up/top_down are re-sorted.

Usage:
  python gate_cache_self_fc.py --cache <vcpe_cache_v3_k562a.json.gz> \
      --expr_h5ad <proc h5ad> --context k562a [--apply]
By default the dry-run only prints the report; with --apply the file is first
backed up into backups_v2_0a/ in the same directory and then written back.

Acceptance criteria (V2-0a go/no-go):
  - k562a: APOC3/ALB/TTR/SERPINA1 self|fc| 0.20~0.60 → ≤0.15; MYC/RPL13A
    unchanged
  - hepg2: ALB (in panel, highly expressed in the liver context) self_fc
    unchanged (differential check)
"""
import argparse
import gzip
import os
import json
import shutil
from pathlib import Path

import anndata as ad
import numpy as np

FLOOR = 0.7                 # panel expression gating threshold (log1p CP10K)
FLAG_NEAR_ZERO = 0.1        # UI warning threshold: only near-zero expression
                            # lights the "low-expression context" badge
                            # (after distribution calibration)
MISMATCH_GATE = 0.15        # suppression factor for Tissue-enriched × lineage
                            # mismatch
# Human Protein Atlas TSV. Previously hard-coded to `parents[4]/rna_platform/...`
# -- a directory OUTSIDE this repository, on the author's machine, with no CLI
# override and no existence check, so this script raised FileNotFoundError for
# every other user. Resolution order is now: --hpa_tsv, then $VCPE_HPA_TSV, then
# data/protein_atlas/proteinatlas.tsv inside the repo. Download it from
# https://www.proteinatlas.org/about/download (proteinatlas.tsv.zip).
DEFAULT_HPA_TSV = (Path(__file__).resolve().parents[2]
                   / "data" / "protein_atlas" / "proteinatlas.tsv")

# Context lineage keywords (substring match against HPA tissue names)
CONTEXT_LINEAGE = {
    "k562a": ("bone marrow", "blood", "lymphoid", "spleen", "thymus"),
    "k562g": ("bone marrow", "blood", "lymphoid", "spleen", "thymus"),
    "jurkat": ("bone marrow", "blood", "lymphoid", "spleen", "thymus"),
    "hepg2": ("liver",),
}


def load_panel_expr(h5ad_fp):
    """Pseudobulk expression of the context ctrl cells (gene -> mean log1p(CP10K))."""
    a = ad.read_h5ad(h5ad_fp, backed="r")
    obs = a.obs
    if "control" in obs.columns:
        mask = obs["control"].astype(int) == 1
    else:
        mask = obs["condition"].astype(str).str.lower() == "ctrl"
    idx = np.where(mask.values)[0]
    sub = a[idx].to_memory()
    X = sub.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    expr = X.mean(axis=0)
    genes = list(sub.var_names)
    sub.file.close()
    a.file.close()
    return dict(zip(genes, map(float, expr)))


def resolve_hpa_tsv(explicit=None):
    """Locate the HPA TSV, or fail with an actionable message."""
    for cand in (explicit, os.environ.get("VCPE_HPA_TSV"), DEFAULT_HPA_TSV):
        if cand and Path(cand).is_file():
            return Path(cand)
    raise FileNotFoundError(
        "Human Protein Atlas TSV not found. Pass --hpa_tsv, set $VCPE_HPA_TSV, "
        f"or place proteinatlas.tsv at {DEFAULT_HPA_TSV}. Download it from "
        "https://www.proteinatlas.org/about/download (proteinatlas.tsv.zip).")


def _column_index(header, name):
    """Match a column whether or not the export quotes its header.

    The released HPA TSV quotes multi-word headers, and this parser used to look
    up the literal string '"RNA tissue specificity"' including the quote
    characters, which breaks the moment the export style changes.
    """
    for i, h in enumerate(header):
        if h.strip().strip('"') == name:
            return i
    raise KeyError(
        f"column {name!r} not found in the HPA TSV header; got "
        f"{[h.strip().strip(chr(34)) for h in header][:8]}...")


def load_hpa_tissue_enriched(hpa_tsv=None):
    """Protein Atlas: tissue-enriched gene -> list of enriched tissues."""
    out = {}
    path = resolve_hpa_tsv(hpa_tsv)
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        i_gene = _column_index(header, "Gene")
        i_spec = _column_index(header, "RNA tissue specificity")
        i_ntpm = _column_index(header, "RNA tissue specific nTPM")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= i_ntpm:
                continue
            spec = parts[i_spec].strip('"')
            if spec != "Tissue enriched":
                continue
            ntpm = parts[i_ntpm].strip('"')
            tissues = [seg.split(":")[0].strip().lower()
                       for seg in ntpm.split(";") if ":" in seg]
            out[parts[i_gene]] = tissues
    return out


def lineage_gate(gene, hpa, lineage_kw):
    """Off-panel gene: Tissue enriched, all enriched lineages mismatch → MISMATCH_GATE."""
    tissues = hpa.get(gene)
    if not tissues:
        return 1.0, "off-panel/no-tissue-enriched"
    if any(any(k in t for k in lineage_kw) for t in tissues):
        return 1.0, "off-panel/lineage-match"
    return MISMATCH_GATE, "off-panel/lineage-mismatch"


def gate_cache(cache_fp, expr_h5ad, context, apply=False, hpa_tsv=None):
    cache_fp = Path(cache_fp)
    with gzip.open(cache_fp, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    genes = cache["genes"]
    lineage_kw = CONTEXT_LINEAGE.get(context, CONTEXT_LINEAGE["k562a"])

    print(f"[expr] loading ctrl pseudobulk from {Path(expr_h5ad).name} ...")
    panel_expr = load_panel_expr(expr_h5ad)
    print(f"[expr] panel genes with ctrl mean: {len(panel_expr)}")
    hpa = load_hpa_tissue_enriched(hpa_tsv)
    print(f"[hpa]  tissue-enriched genes: {len(hpa)}")

    report = {"panel_gated": [], "lineage_gated": [], "unchanged_self": 0}
    expr_flags = {}
    for gene, entry in genes.items():
        # Expression flag (recorded whether or not there is a self-entry; used
        # for the "low-expression context" warning on platform queries)
        e = panel_expr.get(gene)
        if e is not None:
            if e < FLAG_NEAR_ZERO:
                expr_flags[gene] = {
                    "level": "near_zero",
                    "detail": f"ctrl expression log1p={e:.2f} "
                              f"(near-zero expression in this context)",
                    "gate": round(min(1.0, e / FLOOR), 3),
                }
        else:
            lg, _src = lineage_gate(gene, hpa, lineage_kw)
            if lg < 0.999:
                expr_flags[gene] = {
                    "level": "lineage_mismatch",
                    "detail": "tissue-enriched gene (Protein Atlas Tissue "
                              "enriched) does not match this context's lineage",
                    "gate": lg,
                }
        tu, td = entry.get("top_up", []), entry.get("top_down", [])
        if not tu and not td:
            continue
        # self-entry: the gene name in the value list == the target gene itself
        e = panel_expr.get(gene)
        if e is not None:
            gate = min(1.0, e / FLOOR)
            src = f"panel expr={e:.3f}"
        else:
            gate, src = lineage_gate(gene, hpa, lineage_kw)
        if gate >= 0.999:
            report["unchanged_self"] += 1
            continue
        changed = False
        for lst in (tu, td):
            for i, (gn, v) in enumerate(lst):
                if gn == gene:
                    lst[i] = [gn, round(v * gate, 6)]
                    changed = True
        if changed:
            # re-sort (top_up descending / top_down ascending) to keep the
            # ranked lists ordered
            entry["top_up"] = sorted(tu, key=lambda x: -x[1])
            entry["top_down"] = sorted(td, key=lambda x: x[1])
            item = (gene, src, gate)
            (report["panel_gated"] if e is not None else report["lineage_gated"]).append(item)

    n_p, n_l = len(report["panel_gated"]), len(report["lineage_gated"])
    print(f"[gate] panel-gated self entries: {n_p} | lineage-gated: {n_l} "
          f"| unchanged/no-self: {report['unchanged_self']}")

    # acceptance set
    print("\n[verify] self-response of the key genes (after gating):")
    for g in ("APOC3", "ALB", "TTR", "SERPINA1", "PCSK9", "MYC", "RPL13A", "HBZ"):
        if g not in genes:
            print(f"  {g:10s} (not in this context's cache)")
            continue
        entry = genes[g]
        for lst, tag in ((entry.get("top_up", []), "up"), (entry.get("top_down", []), "down")):
            for gn, v in lst:
                if gn == g:
                    e = panel_expr.get(g)
                    print(f"  {g:10s} self_fc={v:+.4f} ({tag})  "
                          f"panel_expr={e if e is not None else 'off-panel'}")

    if not apply:
        print("\n[dry-run] nothing written. Add --apply to persist "
              "(with an automatic backup into backups_v2_0a/).")
        return

    cache["self_fc_gating"] = {
        "version": "v2-0a",
        "panel_floor_log1p": FLOOR,
        "mismatch_gate": MISMATCH_GATE,
        "expr_source": str(Path(expr_h5ad).name),
        "lineage": list(lineage_kw),
        "n_panel_gated": n_p,
        "n_lineage_gated": n_l,
        "n_expr_flags": len(expr_flags),
        "note": ("self-response expression gating (post-process); panel genes linear "
                 "ramp vs ctrl pseudobulk, off-panel tissue-enriched lineage mismatch "
                 "suppressed; trans responses untouched"),
    }
    cache["expr_flags"] = expr_flags
    bdir = cache_fp.parent / "backups_v2_0a"
    bdir.mkdir(exist_ok=True)
    bak = bdir / cache_fp.name
    if not bak.exists():
        shutil.copy2(cache_fp, bak)
        print(f"[backup] {bak}")
    with gzip.open(cache_fp, "wt", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[write] {cache_fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--expr_h5ad", required=True)
    p.add_argument("--context", required=True, choices=list(CONTEXT_LINEAGE))
    p.add_argument("--apply", action="store_true")
    p.add_argument("--hpa_tsv", default=None,
                   help="Human Protein Atlas proteinatlas.tsv; "
                        "defaults to $VCPE_HPA_TSV then "
                        "data/protein_atlas/proteinatlas.tsv")
    args = p.parse_args()
    gate_cache(args.cache, args.expr_h5ad, args.context, apply=args.apply,
               hpa_tsv=args.hpa_tsv)
