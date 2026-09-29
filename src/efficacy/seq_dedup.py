#!/usr/bin/env python
"""Near-duplicate analysis for ASO sequence sets, and split-leakage measurement.

Why exact de-duplication is not enough
--------------------------------------
ASO patent corpora are built by two processes that both defeat exact matching:

  * **Family republication.** A patent family re-files the same sequences across
    members, so one physical oligonucleotide appears many times under different
    patent numbers and different source tables.
  * **Target walking.** A patent tiles a target transcript with a ladder of
    oligonucleotides offset by one nucleotide at a time. Two such sequences are
    different strings but share 15 of 16 positions, target the same site, and
    have near-identical measured potency.

Under either process a grouped split can look clean at the string level while the
held-out rows remain trivially predictable from the training rows. Reporting only
an exact-match overlap count therefore gives a **lower bound** on leakage, and
the repository previously reported nothing at all.

What this measures
------------------
Two sequences are near-duplicates at threshold `t` when the longer one contains
an exact substring of the shorter of length >= `t * len(shorter)`. For tiling
ladders and for length-varied re-files this is the relevant relation: it catches
a shift of any size and a trim at either end, which edit distance on the whole
string does not (a 1-nt shift has edit distance 2 but is the same site).

Candidate pairs come from a k-mer inverted index, so the cost is driven by
genuinely similar pairs rather than by n^2. Comparisons are confined to rows
sharing a target gene by default: a shared substring across different targets is
biologically unremarkable.

Outputs
-------
  * exact-duplicate group count and the size of the largest group
  * near-duplicate pair count at the requested threshold
  * for each candidate grouping column, how many near-duplicate pairs SPAN a
    group boundary -- i.e. how many would survive a grouped split and leak

The last number is the one that matters. A grouping column only controls leakage
if its cross-group near-duplicate count is small.

Usage
-----
    python src/efficacy/seq_dedup.py \
        --parquet data/aso_atlas/aso_atlas_clean.parquet \
        --seq_col aso_sequence_5_to_3 --target_col target_gene \
        --group_cols custom_id patent_number \
        --threshold 0.9 --out_json dedup_report.json
"""
import argparse
import json
from collections import Counter, defaultdict


def kmers(s, k):
    return {s[i:i + k] for i in range(len(s) - k + 1)} if len(s) >= k else {s}


def is_near_duplicate(a, b, threshold):
    """True when the shorter sequence's longest exact substring inside the longer
    one covers at least `threshold` of its length."""
    if len(a) > len(b):
        a, b = b, a
    need = max(1, int(round(threshold * len(a))))
    if need > len(b):
        return False
    # try the longest windows first; the first hit settles it
    for L in range(len(a), need - 1, -1):
        for i in range(0, len(a) - L + 1):
            if a[i:i + L] in b:
                return True
    return False


def find_near_duplicates(seqs, threshold=0.9, k=12, max_candidates_per_seq=2000):
    """Return the list of near-duplicate index pairs among `seqs`.

    `k` sets the inverted-index granularity: two sequences sharing no k-mer
    cannot share a substring of length >= k, so with k <= threshold * min_len the
    index is a sound filter rather than a heuristic.
    """
    index = defaultdict(list)
    for i, s in enumerate(seqs):
        for km in kmers(s, k):
            index[km].append(i)
    pairs, seen = [], set()
    for i, s in enumerate(seqs):
        cand = Counter()
        for km in kmers(s, k):
            bucket = index[km]
            if len(bucket) > max_candidates_per_seq:
                continue                      # ubiquitous k-mer: uninformative
            for j in bucket:
                if j > i:
                    cand[j] += 1
        for j in cand:
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            if is_near_duplicate(s, seqs[j], threshold):
                pairs.append(key)
    return pairs


def analyse(rows, threshold=0.9, k=12, group_cols=(), by_target=True):
    """rows: list of dicts with keys 'seq', 'target', and one per group column."""
    seqs = [r["seq"] for r in rows]

    exact = defaultdict(list)
    for i, s in enumerate(seqs):
        exact[s].append(i)
    dup_groups = {s: idx for s, idx in exact.items() if len(idx) > 1}

    # near-duplicate search, confined to rows sharing a target when asked
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r["target"] if by_target else "_all"].append(i)
    pairs = []
    for _, idx in buckets.items():
        if len(idx) < 2:
            continue
        local = find_near_duplicates([seqs[i] for i in idx], threshold, k)
        pairs.extend((idx[a], idx[b]) for a, b in local)

    out = {
        "n_rows": len(rows),
        "n_unique_sequences": len(exact),
        "n_exact_duplicate_groups": len(dup_groups),
        "n_rows_in_exact_duplicate_groups": sum(len(v) for v in dup_groups.values()),
        "largest_exact_duplicate_group": max((len(v) for v in dup_groups.values()),
                                             default=0),
        "threshold": threshold,
        "kmer_k": k,
        "confined_to_shared_target": by_target,
        "n_near_duplicate_pairs": len(pairs),
    }

    per_group = {}
    for col in group_cols:
        vals = [str(r.get(col)) for r in rows]
        crossing = sum(1 for a, b in pairs if vals[a] != vals[b])
        exact_crossing = 0
        for _, idx in dup_groups.items():
            if len({vals[i] for i in idx}) > 1:
                exact_crossing += 1
        per_group[col] = {
            "n_groups": len(set(vals)),
            "near_duplicate_pairs_crossing_groups": crossing,
            "fraction_of_near_duplicate_pairs_crossing": (
                round(crossing / len(pairs), 4) if pairs else None),
            "exact_duplicate_groups_spanning_groups": exact_crossing,
            "verdict": ("controls near-duplicate leakage"
                        if pairs and crossing / len(pairs) < 0.05
                        else "does NOT control near-duplicate leakage"),
        }
    out["by_group_column"] = per_group
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--seq_col", default="aso_sequence_5_to_3")
    p.add_argument("--target_col", default="target_gene")
    p.add_argument("--group_cols", nargs="*", default=["custom_id"])
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--kmer_k", type=int, default=12)
    p.add_argument("--sample", type=int, default=0,
                   help="analyse a random subsample of this many rows (0 = all); "
                        "the full 188k-row set is feasible but slow")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all_targets", action="store_true",
                   help="do not confine comparisons to rows sharing a target gene")
    p.add_argument("--out_json", default="")
    args = p.parse_args()

    import numpy as np
    import pandas as pd

    df = pd.read_parquet(args.parquet)
    keep = [args.seq_col, args.target_col] + [c for c in args.group_cols if c in df]
    missing = [c for c in args.group_cols if c not in df]
    if missing:
        print(f"[warn] group columns not present and skipped: {missing}", flush=True)
    df = df[keep].dropna(subset=[args.seq_col])
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed)
        print(f"[info] subsampled to {len(df)} rows", flush=True)

    rows = [{"seq": str(r[args.seq_col]).upper(),
             "target": str(r[args.target_col]),
             **{c: r[c] for c in args.group_cols if c in df}}
            for _, r in df.iterrows()]

    rep = analyse(rows, threshold=args.threshold, k=args.kmer_k,
                  group_cols=[c for c in args.group_cols if c in df],
                  by_target=not args.all_targets)

    print(json.dumps(rep, indent=2, default=str), flush=True)
    print("\nInterpretation: a grouping column controls leakage only if its "
          "cross-group near-duplicate count is small. Exact-match overlap alone "
          "is a LOWER BOUND, because patent families republish sequences and "
          "tile targets with single-nucleotide shifts.", flush=True)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(rep, f, indent=2, default=str)
        print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
