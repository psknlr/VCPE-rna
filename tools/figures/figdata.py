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
    p = summary["paired"].get(name, {})
    ci = p.get("ci95_mean_difference") or [None, None]
    return dict(diff=p.get("mean_difference"), lo=ci[0], hi=ci[1],
                verdict=p.get("verdict"), per_seed=p.get("per_seed"))


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
        "P3 head": "P3 head", "zero": "Predict no change",
        "train_mean": "Train mean", "knn_esm2": "ESM2 k-NN",
        "ridge_esm2": "ESM2 ridge", "neighbor_prior": "STRING prior",
        "intact": "Intact", "esm_zero": "Target vector → 0",
        "esm_shuffle": "Target vector permuted", "rna_off": "RNA axis off",
        "is_tgt_off": "is-target off", "is_tgt_shuffle": "is-target permuted",
        "is_nb_off": "is-neighbour off", "is_nb_shuffle": "is-neighbour permuted",
        "ds_shuffle": "Dataset embedding permuted", "all_off": "All conditioning off",
    }.get(name, name)
