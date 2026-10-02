"""Collect every number the figures use, from the committed result files.

No figure in this set may contain a hand-typed value. Hardcoding a measurement
into plotting code is precisely the traceability failure this repository documents
(docs/ERRATA.md E11: committed result files whose schema did not match the script
said to have produced them, so no number could be tied to the code that made it).
Everything here is read from `results/**/*.json`, and `collect()` records which
file each block came from so a figure can be traced back to a run and its
provenance stamp.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(rel):
    p = ROOT / rel
    with open(p) as f:
        return json.load(f)


def _summary(rel):
    """A run_seeds summary -> {model, baselines, paired, seeds, protocol}."""
    d = _load(rel)
    return dict(
        source=rel,
        model=d["model"],
        baselines=d["baselines"],
        paired=d["paired_vs_baselines"],
        seeds=d["seeds"],
        selected_epochs=d.get("selected_epochs"),
        protocol=d.get("protocol", {}),
        estimator=d.get("pearson_dev_estimator"),
    )


def _seed_reports(dirrel, seeds=(0, 1, 2)):
    out = {}
    for s in seeds:
        p = ROOT / dirrel / f"seed{s}" / "final_report.json"
        if p.exists():
            with open(p) as f:
                out[s] = json.load(f)
    return out


def collect():
    d = {}

    # --- the four head-vs-control runs -----------------------------------
    d["plain_35M"] = _summary("results/real_35M/A_seeds/seed_summary.json")
    d["plain_150M"] = _summary("results/real_150M/A_seeds/seed_summary.json")
    d["graph_30ep"] = _summary("results/real_150M_graph/A_seeds/seed_summary.json")
    d["graph_80ep"] = _summary("results/real_150M_graph_long/A_seeds/seed_summary.json")
    d["string_channel"] = _summary("results/real_150M_string/A_seeds/seed_summary.json")

    # per-seed detail, for the robustness figure and the seed-heterogeneity point
    d["seeds_plain_150M"] = _seed_reports("results/real_150M/A_seeds")
    d["seeds_graph_30ep"] = _seed_reports("results/real_150M_graph/A_seeds")
    d["seeds_graph_80ep"] = _seed_reports("results/real_150M_graph_long/A_seeds")

    # --- budget / convergence trajectory (seed 0 only, labelled as such) --
    d["budget_seed0"] = {
        30: _load("results/real_150M_graph/A_seeds/seed0/final_report.json"),
        80: _load("results/real_150M_graph_long/A_seeds/seed0/final_report.json"),
        150: _load("results/real_150M_graph_probe/seed0/final_report.json"),
    }

    # --- mechanism -------------------------------------------------------
    d["ablation_plain"] = _load("results/real_150M/C_ablation.json")
    d["ablation_string"] = _load("results/real_150M_string/C_ablation.json")
    d["sweep"] = _load("results/feature_sweep_innerval.json")

    # --- methodological diagnostics --------------------------------------
    d["e6_masked"] = _load("results/real_150M/A_seeds/seed0/final_report.json")
    d["e6_unmasked"] = _load("results/real_150M/B_nomask/final_report.json")

    # --- external baselines (Tian only) ----------------------------------
    d["table_D"] = _load("results/real_35M/table_D.json")
    d["gears"] = _load("results/real_35M/D_gears.json")
    d["cpa"] = _load("results/real_35M/D_cpa.json")
    d["head_tian"] = _load("results/real_35M/D_head/final_report.json")

    return d


# ---------------------------------------------------------------- helpers

def bar_series(summary, names=("zero", "train_mean", "knn_esm2", "ridge_esm2")):
    """(labels, means, sds) for the model plus the named controls, model first."""
    labels = ["P3 head"] + list(names)
    means = [summary["model"]["mean"]]
    sds = [summary["model"].get("sd")]
    for n in names:
        b = summary["baselines"].get(n, {})
        means.append(b.get("mean"))
        sds.append(b.get("sd"))
    return labels, means, sds


def paired(summary, name):
    """Paired head-minus-control difference, interval recomputed from the splits.

    The committed summaries stored mean +/- 1.96 se, which with three splits is
    a 95% interval only in name (ERRATA E19). The interval is recomputed here
    from the per-split differences with the t quantile, and the stored mean is
    checked against them so a figure cannot silently mix two sources.
    """
    import numpy as np
    from scipy import stats
    p = summary["paired"].get(name, {})
    d = np.asarray(p.get("per_seed") or [], dtype=float)
    if d.size < 2:
        return dict(diff=p.get("mean_difference"), lo=None, hi=None, p=None,
                    n=int(d.size), verdict=p.get("verdict"),
                    per_seed=p.get("per_seed"))
    m = float(d.mean())
    assert abs(m - p["mean_difference"]) < 1e-12, (name, m, p["mean_difference"])
    se = float(d.std(ddof=1)) / np.sqrt(d.size)
    hw = float(stats.t.ppf(0.975, d.size - 1)) * se
    pv = float(2 * stats.t.sf(abs(m / se), d.size - 1)) if se > 0 else None
    return dict(diff=m, lo=m - hw, hi=m + hw, p=pv, n=int(d.size),
                wins=int((d > 0).sum()), verdict=p.get("verdict"),
                per_seed=[float(x) for x in d],
                stored_normal_ci=p.get("ci95_mean_difference"))


def ablation_rows(abl):
    """[(axis, r_vs_real, pearson_dev, delta)] in the script's own order."""
    rep = abl["report"]
    out = []
    for k, v in rep.items():
        if isinstance(v, dict) and "pearson_dev" in v:
            out.append((k, v["r_vs_real"], v["pearson_dev"], v["delta_vs_intact"]))
    return out


def pretty(name):
    return {
        "P3 head": "Conditioned head", "zero": "No change",
        "train_mean": "Training mean", "knn_esm2": "k-NN",
        "ridge_esm2": "Ridge regression", "neighbor_prior": "Neighbour prior",
        "intact": "Intact", "esm_zero": "Target embedding → 0",
        "esm_shuffle": "Target embedding permuted",
        "rna_off": "RNA-sequence block off",
        "is_tgt_off": "Target indicator off",
        "is_tgt_shuffle": "Target indicator permuted",
        "is_nb_off": "Partner indicator off",
        "is_nb_shuffle": "Partner indicator permuted",
        "ds_shuffle": "Screen embedding permuted", "all_off": "All conditioning off",
    }.get(name, name)


# ------------------------------------------------- Extended Data loaders

#: The five configurations run under the corrected protocol, in the order the
#: paper introduces them. Each is (key, label, results directory, epochs).
CONFIGS = [
    ("plain_35M", "ESM2-35M", "results/real_35M/A_seeds", 30),
    ("plain_150M", "ESM2-150M", "results/real_150M/A_seeds", 30),
    ("string_channel", "+ STRING partner indicator", "results/real_150M_string/A_seeds", 30),
    ("graph_30ep", "+ STRING neighbourhood, 30 ep", "results/real_150M_graph/A_seeds", 30),
    ("graph_80ep", "+ STRING neighbourhood, 80 ep", "results/real_150M_graph_long/A_seeds", 80),
]


def train_log(rel):
    """Per-epoch records of the LAST run in a train_log.jsonl, plus its final record.

    Two kinds of boundary are honoured. Newer logs carry an explicit `run_start`
    record; older ones were simply appended to, so a killed run that was resumed
    reads 1..74 then 1..80 with nothing between. Both are split, and the last
    segment is returned, which is the run whose checkpoint was scored.
    """
    recs = [json.loads(l) for l in open(ROOT / rel) if l.strip()]
    starts = [i for i, r in enumerate(recs) if "run_start" in r]
    if starts:
        recs = recs[starts[-1] + 1:]
    epochs, final, seg = [], None, []
    for r in recs:
        if "epoch" in r:
            if seg and r["epoch"] <= seg[-1]["epoch"]:
                seg = []                       # epoch numbering restarted
            seg.append(r)
        elif "final" in r:
            final = r["final"]
    epochs = seg
    n = [r["epoch"] for r in epochs]
    assert n == list(range(1, len(n) + 1)), f"{rel}: epochs not contiguous"
    return dict(source=rel, epochs=epochs, final=final)


def collect_ed():
    d = collect()
    d["configs"] = CONFIGS
    d["reports"] = {k: _seed_reports(path) for k, _, path, _ in CONFIGS}
    d["logs"] = {k: {s: train_log(f"{path}/seed{s}/train_log.jsonl")
                     for s in (0, 1, 2)} for k, _, path, _ in CONFIGS}
    d["log_probe150"] = train_log("results/real_150M_graph_probe/seed0/train_log.jsonl")
    d["sim"] = _load("results/simulation/protocol_traps.json")
    d["d_head"] = _load("results/real_35M/D_head/final_report.json")
    return d
