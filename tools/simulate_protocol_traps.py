#!/usr/bin/env python3
"""Four ways a perturbation-response score rises without any skill, simulated.

Each trap below is a protocol choice this repository changed after finding it
inflated a reported number (docs/ERRATA.md). On real data each was measured
once, at whatever strength the available screens happened to have -- the masking
A/B, for instance, ran on a panel only 0.14% unmeasured, which can bound the
effect from below and nothing more. Here the strength of each trap is swept, and
the score is computed by the repository's OWN metric and baseline code
(`eval_metrics`, `baselines.baseline_train_mean`), so the curves describe the
functions that produced the reported numbers rather than re-implementations.

Every predictor scored here carries, by construction, either no
perturbation-specific information or a fixed amount of it. A score that rises
along a sweep is therefore artefact, and its size is the size of the trap.

  shared     raw fold change contains a response common to every perturbation;
             a predictor that knows nothing about the perturbation (the training
             mean) correlates with it. Scoring the residual removes the credit.
  unmeasured (ERRATA E6) a screen that never measured a gene stores fc = 0, so
             its residual there is a per-screen constant; scored unmasked, the
             training mean is rewarded for reproducing that constant.
  pooled     (ERRATA E4) one Pearson r over the flattened matrix rewards
             predicting each perturbation's overall level; the within-
             perturbation r does not.
  ties       (ERRATA E16) top-k overlap with ties broken by column position
             rewards a constant prediction whenever the panel is ordered by
             variance; the tie-aware form gives it exactly k/n.

Run:  python3 tools/simulate_protocol_traps.py   (about a minute on one CPU)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "maprna_p3"))
sys.path.insert(0, str(ROOT / "src"))
from eval_metrics import (per_item_correlation, pooled_correlation,  # noqa: E402
                          top_k_overlap, top_k_overlap_index_ties)
from baselines import baseline_train_mean                         # noqa: E402
from provenance import write_json                                  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_json", default=str(ROOT / "results" / "simulation" /
                                             "protocol_traps.json"))
    p.add_argument("--n_genes", type=int, default=500)
    p.add_argument("--n_train", type=int, default=240)
    p.add_argument("--n_test", type=int, default=60)
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--informative_r", type=float, default=0.3,
                   help="within-perturbation r of the informative predictor's "
                        "perturbation-specific part, held fixed along the sweep")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def gene_scales(rng, n, shape):
    """Per-gene response s.d., mean 1, sorted descending: a variance-ranked panel,
    which is how make_hvg_list orders the real one. shape=inf gives equal s.d."""
    if np.isinf(shape):
        return np.ones(n)
    return np.sort(rng.gamma(shape, 1.0 / shape, size=n))[::-1]


def summarise(rows):
    a = np.asarray(rows, dtype=np.float64)          # [replicates, grid]
    return dict(mean=a.mean(axis=0).tolist(), sd=a.std(axis=0, ddof=1).tolist())


def trap_shared(args, rng_master):
    lam = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    out = {k: [] for k in ("no_info_raw", "no_info_residual",
                           "informative_raw", "informative_residual")}
    rho = args.informative_r
    for _ in range(args.replicates):
        rng = np.random.default_rng(rng_master.integers(2**63))
        sd = gene_scales(rng, args.n_genes, 1.0)
        row = {k: [] for k in out}
        for l in lam:
            shared = l * sd * rng.normal(size=args.n_genes)
            n = args.n_train + args.n_test
            spec = sd * rng.normal(size=(n, args.n_genes))
            fc = shared + spec
            tr, te = fc[:args.n_train], fc[args.n_train:]
            core = tr.mean(axis=0)                       # train-only common core
            no_info = np.tile(core, (args.n_test, 1))
            noise = sd * rng.normal(size=(args.n_test, args.n_genes))
            informative = core + rho * spec[args.n_train:] + np.sqrt(1 - rho**2) * noise
            row["no_info_raw"].append(per_item_correlation(te, no_info))
            row["no_info_residual"].append(per_item_correlation(te - core, no_info - core))
            row["informative_raw"].append(per_item_correlation(te, informative))
            row["informative_residual"].append(
                per_item_correlation(te - core, informative - core))
        for k in out:
            out[k].append(row[k])
    share = [l**2 / (1 + l**2) for l in lam]
    return dict(x=share, x_param=lam,
                x_label="share of each gene's variance that is common to every "
                        "perturbation",
                series={k: summarise(v) for k, v in out.items()},
                informative_r=rho)


def trap_unmeasured(args, rng_master):
    phi = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    out = {k: [] for k in ("unmasked", "masked")}
    for _ in range(args.replicates):
        rng = np.random.default_rng(rng_master.integers(2**63))
        sd = gene_scales(rng, args.n_genes, 1.0)
        row = {k: [] for k in out}
        for f in phi:
            n = args.n_train + args.n_test
            screen = (rng.random(n) < 0.75).astype(int)  # 1: the partial panel
            shared = sd * rng.normal(size=args.n_genes)
            fc = shared + sd * rng.normal(size=(n, args.n_genes))
            unmeasured = rng.permutation(args.n_genes)[:int(round(f * args.n_genes))]
            mask = np.ones((n, args.n_genes), dtype=bool)
            mask[np.ix_(screen == 1, unmeasured)] = False
            fc = np.where(mask, fc, 0.0)                # what the loader stores
            tr, te = slice(0, args.n_train), slice(args.n_train, n)
            denom = mask[tr].sum(axis=0)
            core = np.where(denom > 0, (fc[tr] * mask[tr]).sum(axis=0)
                            / np.maximum(denom, 1), 0.0)
            dev = fc - core
            dummy = np.zeros((args.n_test, 1))
            pred_u = baseline_train_mean(dev[tr], None, dummy, mask_tr=None)
            pred_m = baseline_train_mean(dev[tr], None, dummy, mask_tr=mask[tr])
            row["unmasked"].append(per_item_correlation(dev[te], pred_u, None))
            row["masked"].append(per_item_correlation(dev[te], pred_m, mask[te]))
        for k in out:
            out[k].append(row[k])
    return dict(x=phi, x_label="fraction of the panel one screen never measured",
                series={k: summarise(v) for k, v in out.items()},
                partial_screen_share=0.75)


def trap_pooled(args, rng_master):
    sig = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    out = {k: [] for k in ("per_item", "pooled")}
    for _ in range(args.replicates):
        rng = np.random.default_rng(rng_master.integers(2**63))
        row = {k: [] for k in out}
        for s in sig:
            level = s * rng.normal(size=args.n_test)
            true = level[:, None] + rng.normal(size=(args.n_test, args.n_genes))
            # knows each perturbation's overall level, and nothing about which
            # genes respond
            pred = level[:, None] + rng.normal(size=(args.n_test, args.n_genes))
            row["per_item"].append(per_item_correlation(true, pred))
            row["pooled"].append(pooled_correlation(true, pred))
        for k in out:
            out[k].append(row[k])
    return dict(x=sig, x_label="s.d. of the per-perturbation level, relative to "
                               "the within-perturbation s.d.",
                series={k: summarise(v) for k, v in out.items()})


def trap_ties(args, rng_master):
    shapes = [np.inf, 8.0, 4.0, 2.0, 1.0, 0.5]
    out = {k: [] for k in ("index_ties_sorted", "index_ties_shuffled", "tie_aware")}
    k = args.k
    for _ in range(args.replicates):
        rng = np.random.default_rng(rng_master.integers(2**63))
        row = {kk: [] for kk in out}
        for sh in shapes:
            sd = gene_scales(rng, args.n_genes, sh)
            true = sd * rng.normal(size=(args.n_test, args.n_genes))
            const = np.zeros_like(true)                 # "predict no change"
            perm = rng.permutation(args.n_genes)
            row["index_ties_sorted"].append(top_k_overlap_index_ties(true, const, k=k))
            row["index_ties_shuffled"].append(
                top_k_overlap_index_ties(true[:, perm], const, k=k))
            row["tie_aware"].append(top_k_overlap(true, const, k=k))
        for kk in out:
            out[kk].append(row[kk])
    cv = [0.0 if np.isinf(s) else float(1.0 / np.sqrt(s)) for s in shapes]
    return dict(x=cv, x_label="coefficient of variation of the per-gene s.d.",
                series={kk: summarise(v) for kk, v in out.items()},
                k=k, chance=k / args.n_genes)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    res = dict(
        description="Synthetic sweeps of four protocol traps, scored with the "
                    "repository's own metric and baseline functions.",
        config=dict(n_genes=args.n_genes, n_train=args.n_train,
                    n_test=args.n_test, replicates=args.replicates, k=args.k,
                    seed=args.seed),
        summary_statistic="mean and sample s.d. across replicates",
        traps=dict(shared=trap_shared(args, rng),
                   unmeasured=trap_unmeasured(args, rng),
                   pooled=trap_pooled(args, rng),
                   ties=trap_ties(args, rng)),
        caveat="Synthetic data with Gaussian responses and gene scales drawn from "
               "a gamma distribution. The sweeps show how each trap scales; they "
               "are not estimates of its size on any real screen.",
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_json, res, args=args)
    for name, t in res["traps"].items():
        print(f"[{name}] x = {[round(v, 3) for v in t['x']]}")
        for s, v in t["series"].items():
            print(f"    {s:22s} {[round(m, 3) for m in v['mean']]}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
