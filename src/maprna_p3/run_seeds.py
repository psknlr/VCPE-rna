#!/usr/bin/env python
"""Run the P3 pipeline over several seeds and aggregate, with the baselines.

Why
---
Every response-line number published before v4 came from a single run with a
single seed, and no statistical testing existed anywhere in the repository. On
this data scale that is not enough to support a comparison: the per-fold spread
already seen on the efficacy line was +/-0.08, which swallows most of the
improvements that were reported as findings.

This runner varies the seed, which changes BOTH the initialisation and the
grouped split (train_p3 derives the split from `--seed`). That is the right unit
of variation for the claim being made: "this model generalises to held-out target
genes" has to survive a different choice of which genes are held out, not merely
a different initialisation. It is a wider interval than an
initialisation-only one, and it is the honest one.

What it reports
---------------
For the model and for every control baseline, across seeds:

    mean, sample sd, min-max, and a normal-approximation 95% CI on the mean

plus the paired per-seed differences model - baseline, which is the quantity a
claim of superiority rests on. Pairing matters: model and baselines share a seed,
hence a split, so the paired difference removes the split-to-split variance that
dominates the marginal spread.

The summary states plainly whether the model beat each baseline on every seed, on
some, or on none.

Usage
-----
    python src/maprna_p3/run_seeds.py --seeds 0 1 2 3 4 \\
        --out_dir seed_runs \\
        -- --data_dirs a.h5ad b.h5ad --esm_table esm.pt --epochs 60 \\
           --n_hvg 2000 --split_by target_gene

Everything after the bare `--` is forwarded verbatim to train_p3.py, except
`--seed` and `--out_dir`, which this script sets per run.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from provenance import write_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", required=True,
                   help="parent directory; each seed gets a subdirectory")
    p.add_argument("--train_script", default=os.path.join(_HERE, "train_p3.py"))
    p.add_argument("--skip_existing", action="store_true",
                   help="reuse a seed's final_report.json if it already exists")
    p.add_argument("--summary_json", default="",
                   help="defaults to <out_dir>/seed_summary.json")
    a, rest = p.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    for banned in ("--seed", "--out_dir"):
        if banned in rest:
            raise SystemExit(f"{banned} is set per run by this script; remove it "
                             f"from the forwarded arguments")
    a.forward = rest
    return a


#: How every interval in a summary is computed. Before ERRATA E19 this was the
#: normal quantile 1.96, which with three splits understates a 95% interval by a
#: factor of t(0.975, 2) / 1.96 = 2.2.
CI_METHOD = "mean +/- t(0.975, n-1) * sd / sqrt(n)"


def t95(n):
    """Two-sided 95% Student-t quantile for a mean of n observations."""
    from scipy import stats
    return float(stats.t.ppf(0.975, n - 1))


def paired_t_p(d):
    """Two-sided paired t-test P value for differences `d` (None if undefined)."""
    from scipy import stats
    d = np.asarray(d, dtype=float)
    if d.size < 2:
        return None
    sd = float(np.std(d, ddof=1))
    if not sd > 0:
        return None
    t = float(d.mean()) / (sd / np.sqrt(d.size))
    return float(2 * stats.t.sf(abs(t), d.size - 1))


def summarise(vals, label):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return dict(label=label, n=0)
    sd = float(np.std(v, ddof=1)) if v.size > 1 else None
    se = (sd / np.sqrt(v.size)) if sd is not None else None
    hw = t95(v.size) * se if se is not None else None
    return dict(label=label, n=int(v.size), mean=float(v.mean()), sd=sd,
                min=float(v.min()), max=float(v.max()),
                ci95_mean=([float(v.mean() - hw), float(v.mean() + hw)]
                           if hw is not None else None),
                ci_method=CI_METHOD,
                per_seed=[float(x) for x in v])


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    reports = {}

    for s in args.seeds:
        d = os.path.join(args.out_dir, f"seed{s}")
        rp = os.path.join(d, "final_report.json")
        if args.skip_existing and os.path.exists(rp):
            print(f"== seed {s}: reusing {rp}", flush=True)
        else:
            cmd = [sys.executable, args.train_script, "--seed", str(s),
                   "--out_dir", d] + args.forward
            print(f"\n== seed {s} ==\n{' '.join(cmd)}", flush=True)
            r = subprocess.run(cmd)
            if r.returncode != 0:
                raise SystemExit(f"seed {s} failed with exit {r.returncode}")
        with open(rp) as f:
            reports[s] = json.load(f)

    seeds = sorted(reports)
    model = [reports[s].get("pearson_dev") for s in seeds]
    base_names = sorted({n for s in seeds
                         for n in reports[s].get("baselines", {})})
    summary = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "variation": ("seed varies both initialisation AND the grouped split, so "
                      "these intervals answer 'does this survive a different "
                      "choice of held-out genes', not merely a different "
                      "initialisation"),
        "model": summarise(model, "model"),
        "baselines": {n: summarise([reports[s]["baselines"].get(n, {}).get("pearson_dev")
                                    for s in seeds], n)
                      for n in base_names},
        "selected_epochs": [reports[s].get("selected_epoch") for s in seeds],
        "split_by": sorted({str(reports[s].get("split_by")) for s in seeds}),
    }

    # One canonical statement of the protocol, taken from the seed reports'
    # own provenance rather than re-derived from this script's passthrough
    # arguments. A consumer that has to parse `forward` to learn the gene panel
    # will eventually parse it differently from the script that used it, which is
    # how a producer and a consumer come to disagree (ERRATA E11).
    PROTOCOL_KEYS = ("data_dirs", "n_hvg", "hvg_from", "split_by", "test_frac",
                     "min_cells", "epochs", "inner_val_frac", "esm_table",
                     "use_mask", "knn_k", "d_model", "inner_val_in_training")
    proto, disagree = {}, {}
    for k in PROTOCOL_KEYS:
        vals = {s: (reports[s].get("provenance", {}).get("args", {}) or {}).get(k)
                for s in seeds}
        uniq = {repr(v) for v in vals.values()}
        if len(uniq) == 1:
            proto[k] = next(iter(vals.values()))
        else:
            disagree[k] = vals
    if disagree:
        raise SystemExit(
            "the seeds were not run under the same protocol, so aggregating them "
            f"would average different experiments: {disagree}")
    summary["protocol"] = proto

    # The estimator name travels with the number or E4 happens again: two
    # definitions of "pearson_dev" were once subtracted from one another. A
    # multi-seed mean that does not say which estimator it averaged cannot be
    # placed beside an external baseline's number.
    ests = {str(reports[s].get("pearson_dev_estimator")) for s in seeds}
    if len(ests) != 1:
        raise SystemExit(f"the seeds used different estimators: {sorted(ests)}. "
                         "Averaging them would repeat ERRATA E4.")
    est = next(iter(ests))
    if est in ("None", ""):
        raise SystemExit("the seed reports do not name their estimator, so this "
                         "mean cannot be compared with anything (ERRATA E4).")
    summary["pearson_dev_estimator"] = est

    # paired differences: model and baselines share a seed, hence a split
    paired = {}
    for n in base_names:
        diffs = []
        for s in seeds:
            m = reports[s].get("pearson_dev")
            b = reports[s].get("baselines", {}).get(n, {}).get("pearson_dev")
            if m is not None and b is not None and np.isfinite(m) and np.isfinite(b):
                diffs.append(m - b)
        d = np.asarray(diffs, dtype=float)
        if d.size == 0:
            paired[n] = dict(n=0)
            continue
        sd = float(np.std(d, ddof=1)) if d.size > 1 else None
        se = (sd / np.sqrt(d.size)) if sd else None
        hw = t95(d.size) * se if se else None
        wins = int((d > 0).sum())
        paired[n] = dict(
            n=int(d.size), mean_difference=float(d.mean()), sd=sd,
            ci95_mean_difference=([float(d.mean() - hw),
                                   float(d.mean() + hw)] if hw else None),
            ci_method=CI_METHOD,
            p_paired_t=paired_t_p(d),
            seeds_where_model_wins=wins,
            verdict=("model beats this baseline on every seed" if wins == d.size
                     else "model loses to this baseline on every seed" if wins == 0
                     else f"mixed: model wins on {wins}/{d.size} seeds"),
            per_seed=[float(x) for x in d])
    summary["paired_vs_baselines"] = paired

    print("\n" + "=" * 72, flush=True)
    m = summary["model"]
    sd_txt = f" +/- {m['sd']:.4f}" if m.get("sd") is not None else ""
    print(f"model pearson_dev: {m['mean']:.4f}{sd_txt} over {m['n']} seeds "
          f"(range {m['min']:.4f}..{m['max']:.4f})", flush=True)
    for n in base_names:
        b, d = summary["baselines"][n], paired[n]
        b_sd = f" +/- {b['sd']:.4f}" if b.get("sd") is not None else ""
        print(f"  vs {n:<12} {b['mean']:.4f}{b_sd} | paired diff "
              f"{d.get('mean_difference', float('nan')):+.4f} | {d.get('verdict')}",
              flush=True)
    losing = [n for n, d in paired.items() if d.get("seeds_where_model_wins", 0) == 0]
    if losing:
        print(f"\nThe model loses on EVERY seed to: {losing}. A conditioned model "
              f"that never beats retrieval or a linear map over the same "
              f"embeddings has not been shown to learn perturbation biology.",
              flush=True)

    out = args.summary_json or os.path.join(args.out_dir, "seed_summary.json")
    write_json(out, summary, args=args)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
