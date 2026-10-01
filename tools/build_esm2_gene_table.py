#!/usr/bin/env python3
"""Build the gene_symbol -> ESM2 embedding table the P3 model is conditioned on.

Why this script exists
----------------------
Every number this repository reports about perturbation response is conditioned
on a table mapping a gene symbol to a protein-language-model vector, and until
now that table existed only on the machine that produced the withdrawn results
(docs/ERRATA.md E10: no external baseline had ever been run, and no real run was
reproducible because the table was never checked in or rebuildable). Nothing can
be regenerated without a table that is derived from public inputs by a recorded
procedure, so this script is the table's definition.

The embedding
-------------
One protein per gene symbol, mean-pooled over the final hidden states of the real
residues (CLS, EOS and padding excluded -- pooling over padding would make the
vector depend on what else happened to be in the batch). Sequences longer than
--max_len are truncated, which is ESM2's own 1024-position limit rather than a
choice; 11% of the human proteome is affected and the truncation is recorded in
the sidecar so it cannot be mistaken for a full-length representation.

Model size
----------
--model selects the ESM2 variant. On a 4-CPU host the 650M variant needs roughly
a day for the human proteome, so a smaller variant is the practical choice; the
variant is written into the sidecar because it is part of what produced every
downstream number. The comparison against the ESM2 retrieval and ridge controls
is unaffected by the choice -- all three read this same table -- but the absolute
scores are not, so the table and the results must be quoted together.

Resumability
------------
Embeddings are flushed to numbered shards as they are computed and re-read on a
restart. This is not a convenience: the run takes hours, and an interrupted run
that had to start over would be an hours-long silent cost.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import write_json  # noqa: E402

# UniProt's protein-existence levels, best first. Used only to pick one protein
# per symbol when several are reviewed; ties then go to the longer sequence.
PE_RANK = {
    "Evidence at protein level": 0,
    "Evidence at transcript level": 1,
    "Inferred from homology": 2,
    "Predicted": 3,
    "Uncertain": 4,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uniprot_tsv", required=True,
                   help="TSV with columns Entry, Gene Names (primary), Sequence, "
                        "Length, Protein existence (UniProt REST, fields="
                        "accession,gene_primary,sequence,length,protein_existence)")
    p.add_argument("--out", required=True, help="output .pt table (symbol -> vector)")
    p.add_argument("--model", default="facebook/esm2_t30_150M_UR50D")
    p.add_argument("--max_len", type=int, default=1022,
                   help="residues kept per protein; ESM2 positions minus CLS/EOS")
    p.add_argument("--token_budget", type=int, default=4096,
                   help="padded residues per batch, i.e. batch_size * the batch's "
                        "LONGEST sequence. It must be the longest and not the first: "
                        "sequences are length-sorted, so sizing a batch from its "
                        "shortest member overshoots the real padded size by more than "
                        "an order of magnitude and gets the process OOM-killed.")
    p.add_argument("--max_batch", type=int, default=32,
                   help="hard cap on batch size, so a run of very short proteins "
                        "cannot build a batch of thousands")
    p.add_argument("--threads", type=int, default=0, help="0 = leave torch's default")
    p.add_argument("--limit_symbols", type=str, default=None,
                   help="optional file with one gene symbol per line: embed only "
                        "these (plus nothing else). Use to cut the run down to the "
                        "genes a dataset actually contains.")
    p.add_argument("--shard_every", type=int, default=500,
                   help="flush a resumable shard after this many proteins")
    p.add_argument("--limit", type=int, default=0, help="debug: stop after N symbols")
    return p.parse_args()


def load_proteome(tsv, limit_symbols=None):
    """One sequence per gene symbol, chosen deterministically."""
    want = None
    if limit_symbols:
        want = {l.strip() for l in open(limit_symbols) if l.strip()}
    best = {}
    with open(tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for col in ("Entry", "Gene Names (primary)", "Sequence", "Protein existence"):
            if col not in idx:
                raise SystemExit(f"{tsv}: missing column {col!r}; got {header}")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < len(header):
                continue
            sym = p[idx["Gene Names (primary)"]].strip()
            seq = p[idx["Sequence"]].strip()
            if not sym or not seq:
                continue
            # UniProt returns several symbols for a few entries; the first is primary
            sym = sym.split(";")[0].split()[0] if sym else ""
            if not sym or (want is not None and sym not in want):
                continue
            key = (PE_RANK.get(p[idx["Protein existence"]].strip(), 9), -len(seq),
                   p[idx["Entry"]])
            if sym not in best or key < best[sym][0]:
                best[sym] = (key, p[idx["Entry"]], seq)
    return {s: (acc, seq) for s, (_, acc, seq) in best.items()}


def shard_paths(out):
    d = Path(out).parent / (Path(out).stem + "_shards")
    return d


def load_shards(out):
    d = shard_paths(out)
    done = {}
    if not d.is_dir():
        return done
    for f in sorted(d.glob("shard_*.pt")):
        try:
            done.update(torch.load(f, map_location="cpu", weights_only=False))
        except Exception as e:  # a shard cut off by a restart is dropped, not fatal
            print(f"[esm2] ignoring unreadable shard {f.name}: {e}", flush=True)
    return done


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    from transformers import AutoModel, AutoTokenizer

    prot = load_proteome(args.uniprot_tsv, args.limit_symbols)
    syms = sorted(prot)
    if args.limit:
        syms = syms[:args.limit]
    done = load_shards(args.out)
    todo = [s for s in syms if s not in done]
    n_trunc = sum(1 for s in syms if len(prot[s][1]) > args.max_len)
    print(f"[esm2] symbols={len(syms)} already_embedded={len(done)} todo={len(todo)} "
          f"truncated_at_{args.max_len}={n_trunc} ({n_trunc / max(1, len(syms)) * 100:.1f}%)",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModel.from_pretrained(args.model, dtype=torch.float32).eval()
    dim = int(mdl.config.hidden_size)

    # length-sorted so a batch pads to nearly its own longest member
    todo.sort(key=lambda s: len(prot[s][1]))
    d_shard = shard_paths(args.out)
    d_shard.mkdir(parents=True, exist_ok=True)
    pending, t0, n_done = {}, time.time(), 0
    i = 0
    while i < len(todo):
        # Grow the batch while batch_size * longest_member stays inside the budget.
        # The padded tensor is that product, not batch_size * first_member.
        batch, Lmax = [], 0
        j = i
        while j < len(todo) and len(batch) < args.max_batch:
            Lj = min(len(prot[todo[j]][1]), args.max_len) + 2
            if batch and max(Lmax, Lj) * (len(batch) + 1) > args.token_budget:
                break
            batch.append(todo[j]); Lmax = max(Lmax, Lj); j += 1
        seqs = [prot[s][1][:args.max_len] for s in batch]
        enc = tok(seqs, return_tensors="pt", padding=True)
        with torch.no_grad():
            h = mdl(**enc).last_hidden_state                    # [B, T, d]
        # mean over real residues only: drop CLS, EOS and padding
        am = enc["attention_mask"].clone()
        am[:, 0] = 0                                            # CLS
        for b, s in enumerate(seqs):
            am[b, len(s) + 1] = 0                               # EOS
        w = am.unsqueeze(-1).to(h.dtype)
        vec = (h * w).sum(1) / w.sum(1).clamp(min=1.0)
        for b, s in enumerate(batch):
            pending[s] = vec[b].to(torch.float32).clone()
        i += len(batch)
        n_done += len(batch)
        if len(pending) >= args.shard_every or i >= len(todo):
            k = len(list(d_shard.glob("shard_*.pt")))
            tmp = d_shard / f".tmp_shard_{k:05d}.pt"
            torch.save(pending, tmp)
            os.replace(tmp, d_shard / f"shard_{k:05d}.pt")      # atomic: no half shard
            done.update(pending)
            pending = {}
            rate = n_done / max(1e-9, time.time() - t0)
            left = (len(todo) - i) / max(1e-9, rate)
            print(f"[esm2] {i}/{len(todo)} ({rate * 60:.0f}/min, ~{left / 60:.0f} min left)",
                  flush=True)

    table = {s: done[s] for s in syms if s in done}
    missing = [s for s in syms if s not in done]
    if missing:
        raise SystemExit(f"[esm2] {len(missing)} symbols never embedded, e.g. {missing[:5]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.out)
    lens = np.array([len(prot[s][1]) for s in syms])
    write_json(str(Path(args.out).with_suffix(".provenance.json")), dict(
        table=os.path.abspath(args.out), n_symbols=len(table), dim=dim,
        esm2_model=args.model,
        esm2_model_note="Not the 650M variant used for the withdrawn results: 650M "
                        "needs about a day for this proteome on 4 CPUs. The ESM2 "
                        "retrieval and ridge controls read this same table, so the "
                        "comparison between them and the model is unaffected; the "
                        "absolute scores are not, and must be quoted with this table.",
        max_len=args.max_len,
        truncated_fraction=float((lens > args.max_len).mean()),
        pooling="mean over final hidden states of real residues (CLS/EOS/pad excluded)",
        one_protein_per_symbol="best UniProt protein-existence level, then longest "
                               "sequence, then lowest accession",
        uniprot_tsv=os.path.abspath(args.uniprot_tsv),
        seq_len_mean=float(lens.mean()), seq_len_median=float(np.median(lens)),
    ), args=args)
    print(f"[esm2] wrote {args.out}: {len(table)} symbols x {dim}", flush=True)


if __name__ == "__main__":
    main()
