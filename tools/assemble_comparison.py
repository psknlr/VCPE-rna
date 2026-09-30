#!/usr/bin/env python3
"""Assemble one comparison table from the per-model report files, or refuse to.

Why this is a script and not a hand-written table
-------------------------------------------------
Almost every withdrawn number in docs/ERRATA.md was withdrawn for the same
reason: two numbers were put next to each other that were not measuring the same
thing. E4 subtracted a within-perturbation correlation from a pooled one. E3
compared an epoch chosen on the reported slice against one that was not. E1
correlated a feature tensor against a label. None of those is visible in a
finished table -- they are only visible in how each number was produced.

So this refuses to emit a table whose rows disagree about the protocol, and names
the fields that differ. A table it does emit is one in which the split, the gene
panel, the mask, the metric and the estimator were the same for every row.

It performs no computation on the metrics. It reads what each adapter wrote,
checks the rows are comparable, and lays them out.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import write_json  # noqa: E402

# Fields that decide WHAT is being measured. Two rows that differ on any of these
# are not comparable, whatever their numbers look like.
#
# esm_table is one of them, and for a reason worth stating: the head, the k-NN
# retrieval control and the ridge control all read the same table, so the table
# does not bias the comparison BETWEEN them -- but it does move all three, and it
# does not move them equally. The controls are pure functions of the embedding,
# while the head also sees the control profile and the dataset embedding, so a
# weaker table handicaps the controls more than the head. Putting a row from one
# table beside a row from another would therefore flatter the head, in a way no
# amount of care in reading the numbers would reveal. It is compared by path,
# which is a proxy: two tables built from different ESM2 variants to the same path
# would slip through, so the variant is recorded in each table's sidecar.
REQUIRED_SAME = ("data_dirs", "n_hvg", "hvg_from", "split_by", "test_frac",
                 "min_cells", "seed", "use_mask", "esm_table")

# Fields worth showing but which legitimately differ between models.
INFORMATIVE = ("epochs", "train_h5ad", "train_h5ad_is_cell_level", "device")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reports", nargs="+", required=True,
                   help="report JSONs (train_p3 final_report.json, run_seeds "
                        "seed_summary.json, baseline_gears / baseline_cpa out_json). "
                        "Globs are expanded.")
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_md", default=None, help="defaults to --out_json with .md")
    p.add_argument("--metric", default="pearson_dev")
    p.add_argument("--allow", nargs="*", default=[],
                   help="protocol fields allowed to differ. Each one is printed in "
                        "the table's caveats, because relaxing a requirement does not "
                        "make the rows comparable, it only records that you chose to "
                        "compare them anyway.")
    return p.parse_args()


# fields that name one value per run. A report may carry them wrapped in a
# one-element list -- run_seeds used to summarise split_by as the SET of values
# across its seeds -- and comparing a list against the same value unwrapped would
# report a mismatch that does not exist.
SCALAR = ("n_hvg", "hvg_from", "split_by", "test_frac", "min_cells", "epochs",
          "use_mask")


def protocol_of(rep):
    """The protocol fingerprint.

    Preference order: the report's own `protocol` block, then the parsed
    arguments in its provenance, then a top-level key. The first is canonical --
    a consumer that reconstructs the protocol from a passthrough argument list
    will eventually reconstruct it differently from the script that used it.
    """
    prov = rep.get("provenance", {}) or {}
    args = prov.get("args", {}) or {}
    block = rep.get("protocol", {}) or {}
    out = {}
    for k in REQUIRED_SAME + INFORMATIVE:
        v = block.get(k, args.get(k, rep.get(k, (rep.get("config") or {}).get(k))))
        if k == "use_mask" and v is None:
            # Masked and unmasked scoring are different measurements -- that is
            # ERRATA E6 -- so a report that does not say which it did cannot be
            # placed beside one that does.
            raise SystemExit(
                "a report does not record whether the measured-column mask was "
                "applied. Masked and unmasked scoring are different measurements "
                "(ERRATA E6), so they cannot share a table. Re-run it with a "
                "version that records use_mask.")
        if k in SCALAR and isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        if isinstance(v, list):
            v = tuple(os.path.abspath(str(x)) if k == "data_dirs" else x for x in v)
        out[k] = v
    # run_seeds varies the seed on purpose and reports the set it used
    if "seeds" in args and args["seeds"]:
        out["seed"] = tuple(sorted(args["seeds"]))
    elif out.get("seed") is not None:
        out["seed"] = (out["seed"],)
    return out


def rows_of(path, rep, metric):
    """(label, value, n_runs, spread, estimator, kind) for each row a report holds."""
    rows = []
    src = os.path.basename(path)

    # run_seeds summary: model + baselines, each aggregated over seeds
    if "paired_vs_baselines" in rep:
        m = rep.get("model", {})
        rows.append(dict(label="P3 head", value=m.get("mean"), n=m.get("n"),
                         sd=m.get("sd"), lo=m.get("min"), hi=m.get("max"),
                         estimator=rep.get("pearson_dev_estimator"),
                         kind="multi-seed mean", source=src))
        for name, b in (rep.get("baselines") or {}).items():
            pr = (rep.get("paired_vs_baselines") or {}).get(name, {})
            # run_seeds names this key mean_difference, not mean. Reading the
            # wrong one printed "n/a" in the paired column with no error -- the
            # same shape as ERRATA E11's committed-JSON mismatches, so both
            # spellings are accepted and the test fixture uses the producer's.
            pd_ = pr.get("mean_difference", pr.get("mean"))
            rows.append(dict(label=name, value=b.get("mean"), n=b.get("n"),
                             sd=b.get("sd"), lo=b.get("min"), hi=b.get("max"),
                             kind="multi-seed mean", source=src,
                             paired_diff=pd_, paired_verdict=pr.get("verdict")))
        return rows

    # single train_p3 run
    if "baselines" in rep and metric in rep:
        rows.append(dict(label="P3 head", value=rep.get(metric), n=1,
                         estimator=rep.get("pearson_dev_estimator"),
                         kind="single run", source=src))
        for name, b in (rep["baselines"] or {}).items():
            rows.append(dict(label=name, value=b.get(metric), n=1,
                             kind="deterministic control", source=src))
        return rows

    # An external adapter. The two adapters do not share a schema, so both are
    # handled: CPA writes a single `cpa` block with the metric directly; GEARS,
    # which repeats runs because it is not deterministic at a fixed seed, writes
    # `gears_across_runs` with metric-prefixed keys (pearson_dev_mean/_sd/_min/
    # _max) and a `gears_per_run` list. Reading only the block form is how this
    # first refused a real GEARS report -- the same producer/consumer mismatch as
    # ERRATA E11, found because no test ran a real adapter report through here.
    for key in ("gears", "cpa", "scgpt"):
        across = rep.get(f"{key}_across_runs")
        per_run = rep.get(f"{key}_per_run")
        blk = rep.get(key) if isinstance(rep.get(key), dict) else None
        if across is None and per_run is None and blk is None:
            continue
        if across:
            value = across.get(f"{metric}_mean", across.get("mean"))
            sd = across.get(f"{metric}_sd", across.get("sd"))
            lo = across.get(f"{metric}_min", across.get("min"))
            hi = across.get(f"{metric}_max", across.get("max"))
            n = across.get("n_runs") or (len(per_run) if per_run else 1)
        elif blk:
            value, sd, lo, hi, n = blk.get(metric), None, None, None, 1
        else:                                    # only per_run present
            vals = [r.get(metric) for r in per_run if r.get(metric) is not None]
            import statistics
            value = statistics.fmean(vals) if vals else None
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            lo, hi, n = (min(vals) if vals else None), (max(vals) if vals else None), len(vals)
        # estimator: from the block, else from a per-run entry
        est = (blk or {}).get("pearson_dev_estimator")
        if est is None and per_run:
            est = per_run[0].get("pearson_dev_estimator")
        rows.append(dict(
            label=key.upper(), value=value, n=n, sd=sd, lo=lo, hi=hi,
            estimator=est,
            kind=("mean over runs" if (n or 0) > 1 else "single run"),
            source=src))
        return rows
    raise SystemExit(f"{path}: cannot tell which model this report describes. "
                     "Expected a run_seeds summary, a train_p3 final_report, or an "
                     "external adapter report.")


def main():
    args = parse_args()
    paths = []
    for pat in args.reports:
        hits = sorted(glob.glob(pat))
        if not hits:
            raise SystemExit(f"no report matched {pat!r}")
        paths += hits

    reports, protos, rows, caveats, estimators = {}, {}, [], [], set()
    for path in paths:
        with open(path) as f:
            rep = json.load(f)
        reports[path] = rep
        # identify the report before judging its protocol: "I cannot tell what
        # this file is" is the more fundamental complaint, and reporting the
        # protocol problem first would send a reader looking in the wrong place
        rows += rows_of(path, rep, args.metric)
        protos[path] = protocol_of(rep)
        fair = (rep.get("fairness") or {}).get("not_controlled") or []
        caveats += [f"{os.path.basename(path)}: {c}" for c in fair]

    # --- the refusal ---------------------------------------------------------
    allowed = set(args.allow)
    differ = {}
    for field in REQUIRED_SAME:
        vals = {p: protos[p].get(field) for p in paths}
        if len({repr(v) for v in vals.values()}) > 1:
            differ[field] = vals
    hard = {k: v for k, v in differ.items() if k not in allowed}
    if hard:
        print("REFUSING to put these numbers in one table: the rows are not "
              "measuring the same thing.\n", file=sys.stderr)
        for field, vals in hard.items():
            print(f"  {field}:", file=sys.stderr)
            for p, v in vals.items():
                print(f"      {os.path.basename(p):32s} {v}", file=sys.stderr)
        print("\nFix the runs so these agree, or pass --allow <field> to record "
              "that you compared them anyway.", file=sys.stderr)
        raise SystemExit(2)
    for field in differ:
        caveats.append(
            f"PROTOCOL MISMATCH ALLOWED: {field} differs between rows "
            f"({ {os.path.basename(p): v for p, v in differ[field].items()} }). "
            "The rows are not strictly comparable.")
    estimators = {r["estimator"] for r in rows if r.get("estimator")}
    if len(estimators) > 1:
        raise SystemExit(f"two different estimators in one table: {sorted(estimators)}. "
                         "This is ERRATA E4 exactly; refusing.")

    shared = {k: protos[paths[0]].get(k) for k in REQUIRED_SAME if k not in differ}
    rows.sort(key=lambda r: (-(r["value"] if r["value"] is not None else -9e9),))

    # --- emit ---------------------------------------------------------------
    def fmt(v, nd=4):
        return "n/a" if v is None else f"{v:+.{nd}f}"

    lines = [f"# {args.metric} on a shared protocol", "",
             "Every row below was produced with the same "
             + ", ".join(f"`{k}`" for k in shared) + ".", ""]
    lines += [f"- `{k}` = `{v}`" for k, v in shared.items()]
    lines += ["", f"| model | {args.metric} | runs | sd | range | kind |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        rng = "—" if r.get("lo") is None else f"{fmt(r['lo'], 3)}..{fmt(r['hi'], 3)}"
        sd = "—" if r.get("sd") is None else f"{r['sd']:.4f}"
        lines.append(f"| {r['label']} | {fmt(r['value'])} | {r.get('n') or 1} | "
                     f"{sd} | {rng} | {r['kind']} |")
    if any(r.get("paired_verdict") for r in rows):
        lines += ["", "## Paired per-seed comparison against the P3 head", "",
                  "Pairing removes split-to-split variance, which is larger than the "
                  "differences being compared.", "",
                  "| baseline | mean paired difference | verdict |", "|---|---|---|"]
        for r in rows:
            if r.get("paired_verdict"):
                lines.append(f"| {r['label']} | {fmt(r.get('paired_diff'))} | "
                             f"{r['paired_verdict']} |")
    if caveats:
        lines += ["", "## What is not controlled", "",
                  "These come from each adapter's own `fairness` block. A number from "
                  "this table quoted without them is misleading.", ""]
        lines += [f"- {c}" for c in dict.fromkeys(caveats)]

    md = "\n".join(lines) + "\n"
    out_md = args.out_md or str(Path(args.out_json).with_suffix(".md"))
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(md)
    write_json(args.out_json, dict(metric=args.metric, shared_protocol=
                                   {k: list(v) if isinstance(v, tuple) else v
                                    for k, v in shared.items()},
                                   rows=rows, caveats=list(dict.fromkeys(caveats)),
                                   estimator=(sorted(estimators) or [None])[0],
                                   reports=[os.path.abspath(p) for p in paths]),
               args=args)
    print(md)
    print(f"[assemble] wrote {args.out_json} and {out_md}")


if __name__ == "__main__":
    main()
