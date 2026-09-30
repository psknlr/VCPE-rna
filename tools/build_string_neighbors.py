#!/usr/bin/env python3
"""Build the STRING neighbour table for the P3 model's `is_neighbor` channel.

Why this exists
---------------
The per-axis ablation on real data (docs/RESULTS_REAL.md) found that three of the
model's four claimed identity channels leave the prediction **bit-identical**:
all of its conditioning flows through the target gene's ESM2 vector alone. Two of
those channels were inert because their inputs were never built -- this is one of
them.

That matters for more than tidiness. The ridge and k-NN controls read the same
ESM2 vectors the head does, so as long as the head sees nothing else, its only
possible advantage is non-linearity on identical features -- which is why the
measured margin is +0.027. `is_neighbor` is different in kind: knocking a gene
down should perturb its interaction partners, and that is information neither
control can express. Whether it actually helps is an empirical question this
table makes askable.

What it writes
--------------
`[V, K]` int32, row i = the ESM2 table row indices of gene i's top-K STRING
partners by combined score, padded with -1. The model reads it as
`is_neighbor[b, h] = hvg_rows[h] in neighbours(pert_rows[b])`.

Fairness note
-------------
A model given a biological prior that its baselines cannot express is a model
given more information, not necessarily a better model. Any gain measured with
this table has to be reported against a graph-only control (`neighbor_prior` in
src/maprna_p3/baselines.py) as well as against the ESM2 controls, or the
comparison attributes to architecture what belongs to the input.
"""
import argparse
import gzip
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import write_json  # noqa: E402

BASE = "https://stringdb-downloads.org/download"
LINKS = f"{BASE}/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
INFO = f"{BASE}/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--esm_table", required=True,
                   help="the gene_symbol -> vector table whose ROW ORDER this "
                        "neighbour table must match; the model indexes both by the "
                        "same row id, so a mismatch silently scrambles the graph")
    p.add_argument("--out", required=True, help="output .npy, [V, K] int32")
    p.add_argument("--cache_dir", default="/home/user/data_real/string",
                   help="where the downloaded STRING files are kept")
    p.add_argument("--min_score", type=int, default=700,
                   help="minimum STRING combined_score (700 = high confidence, "
                        "STRING's own conventional cutoff)")
    p.add_argument("--top_k", type=int, default=32,
                   help="neighbours kept per gene, highest score first")
    return p.parse_args()


def fetch(url, dest):
    """Download once, tolerating a proxy that re-compresses an already-gzipped body."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[string] using cached {os.path.basename(dest)}", flush=True)
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[string] downloading {os.path.basename(dest)} ...", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r:
        raw = r.read()
    # The egress proxy here gzips responses that are already gzip, so peel until
    # the result is no longer a gzip stream rather than trusting Content-Encoding.
    layers = 0
    while raw[:2] == b"\x1f\x8b" and layers < 4:
        try:
            peeled = gzip.decompress(raw)
        except Exception:
            break
        raw, layers = peeled, layers + 1
    with open(dest, "wb") as f:
        f.write(raw)
    print(f"[string]   {len(raw) / 1e6:.0f} MB after {layers} gzip layer(s)", flush=True)
    return dest


def open_maybe_gzip(path):
    with open(path, "rb") as f:
        magic = f.read(2)
    return gzip.open(path, "rt") if magic == b"\x1f\x8b" else open(path, "rt")


def main():
    args = parse_args()
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab)
    sym2row = {s: i for i, s in enumerate(symbols)}
    V = len(symbols)
    print(f"[string] ESM2 table: {V} symbols", flush=True)

    info_p = fetch(INFO, os.path.join(args.cache_dir, "9606.protein.info.txt"))
    links_p = fetch(LINKS, os.path.join(args.cache_dir, "9606.protein.links.txt"))

    # STRING protein id -> gene symbol, keeping only genes the ESM2 table covers
    prot2row = {}
    with open_maybe_gzip(info_p) as f:
        header = f.readline()
        assert "preferred_name" in header, f"unexpected info header: {header[:120]}"
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 2:
                continue
            r = sym2row.get(p[1])
            if r is not None:
                prot2row[p[0]] = r
    print(f"[string] protein ids mapped into the table: {len(prot2row)}", flush=True)
    if len(prot2row) < 0.5 * V:
        print(f"[string] WARNING: only {len(prot2row)}/{V} symbols matched. A low "
              f"match rate means the graph is mostly absent and the channel will "
              f"stay near-inert.", flush=True)

    # top-K partners per gene above the confidence cutoff
    best = defaultdict(list)
    n_edges = n_kept = 0
    with open_maybe_gzip(links_p) as f:
        f.readline()
        for line in f:
            a, b, sc = line.split()
            n_edges += 1
            s = int(sc)
            if s < args.min_score:
                continue
            ra, rb = prot2row.get(a), prot2row.get(b)
            if ra is None or rb is None or ra == rb:
                continue
            n_kept += 1
            best[ra].append((s, rb))
            best[rb].append((s, ra))
    print(f"[string] edges: {n_edges} total, {n_kept} kept at score >= "
          f"{args.min_score} and mappable", flush=True)

    out = np.full((V, args.top_k), -1, dtype=np.int32)
    degrees = np.zeros(V, dtype=np.int32)
    for r, lst in best.items():
        # deterministic: score descending, then row id, so the table does not
        # depend on dict or file ordering
        seen, keep = set(), []
        for s, nb in sorted(lst, key=lambda t: (-t[0], t[1])):
            if nb in seen:
                continue
            seen.add(nb)
            keep.append(nb)
            if len(keep) == args.top_k:
                break
        out[r, :len(keep)] = keep
        degrees[r] = len(keep)

    covered = int((out >= 0).any(axis=1).sum())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out)
    print(f"[string] wrote {args.out}: {out.shape} | genes with >=1 neighbour "
          f"{covered}/{V} ({covered / V * 100:.0f}%) | median degree "
          f"{int(np.median(degrees[degrees > 0])) if covered else 0}", flush=True)
    write_json(str(Path(args.out).with_suffix(".provenance.json")), dict(
        out=os.path.abspath(args.out), shape=list(out.shape),
        esm_table=os.path.abspath(args.esm_table), n_symbols=V,
        string_version="v12.0", organism="9606",
        min_combined_score=args.min_score, top_k=args.top_k,
        n_edges_total=n_edges, n_edges_kept=n_kept,
        protein_ids_mapped=len(prot2row),
        genes_with_neighbours=covered,
        coverage_fraction=covered / V,
        median_degree=(int(np.median(degrees[degrees > 0])) if covered else 0),
        row_order="identical to the ESM2 table's insertion order; both are indexed "
                  "by the same row id and a mismatch would scramble the graph",
        tie_break="score descending, then row id, so the table is independent of "
                  "file and dict ordering",
        fairness_note="A gain measured with this table must be reported against a "
                      "graph-only control as well as the ESM2 controls, or it "
                      "attributes to architecture what belongs to extra input.",
    ), args=args)


if __name__ == "__main__":
    main()
