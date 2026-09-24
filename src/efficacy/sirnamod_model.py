# -*- coding: utf-8 -*-
"""siRNAmod (Martinelli 2023, 907 modified siRNAs) potency model.

Task   : HELM modification pattern + sequence -> percent inhibition (0-100)
Method : XGBoost, nested 5-fold CV grouped by parent duplex (CPU, <1 min)

Features (538 total, as verified against the shipped artifact):
  L1 global   : per-strand count of each of the 8 modification classes -> 16
  L2 positional: per-position modification one-hot (21 x 2 x 8 = 336)
                 + per-position base one-hot (21 x 2 x 4 = 168)       -> 504
  L3 regional : per-modification counts in the antisense seed region
                (`antisense[1:8]`, i.e. positions 2-8) and in the cleavage
                region (`antisense[9:12]`, i.e. positions 10-12)       -> 16
  GC content                                                            -> 2

  These are REGION-RESTRICTED COUNTS, not crossed terms: there is no explicit
  modification x base or modification x position interaction feature. Earlier
  documentation described them as "region-crossed" and gave the feature counts
  as 12 / 252 / 168, none of which match the implementation.

  Caveat: for duplexes shorter than 12 nt the L3 block is not emitted at all and
  the downstream `.get(k, 0)` fills it with zeros, so "feature absent" and
  "count is zero" are indistinguishable. The same holds for the high-index
  one-hot columns of short strands.

Evaluation:
  907 rows over 538 features is 1.7 rows per feature, so the split and the
  stopping rule decide the number. Outer folds are grouped by parent duplex,
  because the dataset is a few parent duplexes crossed with many modification
  patterns and 504 of the features identify the parent directly; the tree
  count is chosen by early stopping on a grouped inner split rather than fixed
  by hand. Run with --also_random_cv to see what the previous ungrouped random
  KFold reported for comparison.
"""
import gzip
import csv
import re
import collections
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "oligogym")
MOD_ORDER = ['unmod', 'fl2r', 's4r', 'lna', 'hna', 'una', 'ome', 'dna']


def parse_helm_strand(units):
    """Parse a list of HELM unit tokens into (base, mod) tuples, excluding phosphates."""
    nts = []
    for unit in units:
        if unit.rstrip('.') in ('p', 'p.'):
            continue
        base_m = re.search(r'\(([A-Z])\)', unit)
        base = base_m.group(1) if base_m else '?'
        # detect modification
        bracket_mods = re.findall(r'\[([^\]]+)\]', unit)
        if bracket_mods:
            mod = bracket_mods[0]
        elif unit.startswith('m('):
            mod = 'ome'
        elif unit.startswith('r(') or unit.startswith('[r]'):
            mod = 'unmod'
        elif unit.startswith('d('):
            mod = 'dna'
        else:
            mod = 'unmod'
        nts.append((base, mod))
    return nts


def parse_helm(helm):
    """Parse full HELM string -> (sense_nts, antisense_nts)."""
    strands_raw = helm.split('|')
    parsed = []
    for s in strands_raw:
        m = re.search(r'\{(.+)\}', s)
        if not m:
            parsed.append([])
            continue
        content = m.group(1)
        units = re.findall(r'(\[?[a-z0-9]+\]?\([A-Z]\)|p\b\.?)', content)
        parsed.append(parse_helm_strand(units))
    # strand 0 = sense, strand 1 = antisense
    sense = parsed[0] if len(parsed) > 0 else []
    antisense = parsed[1] if len(parsed) > 1 else []
    return sense, antisense


def build_features(sense, antisense, max_len=25):
    """Build feature vector for one siRNA."""
    feats = {}
    # L1: global mod counts per strand
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        mod_counts = collections.Counter(m for _, m in nts)
        for m in MOD_ORDER:
            feats[f'{strand_name}_n_{m}'] = mod_counts.get(m, 0)
    # L2: positional mod one-hot
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for m in MOD_ORDER:
                feats[f'{strand_name}{pos}_{m}'] = 1 if mod == m else 0
    # sequence one-hot
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for b in 'ACGU':
                feats[f'{strand_name}{pos}_{b}'] = 1 if base == b else 0
    # L3: functional region indicators
    # seed region = antisense positions 1-7 (0-indexed), cleavage site = 9-11
    if len(antisense) >= 12:
        for m in MOD_ORDER:
            seed_count = sum(1 for pos, (b, mo) in enumerate(antisense[1:8]) if mo == m)
            feats[f'Aseed_{m}'] = seed_count
            cleav_count = sum(1 for pos, (b, mo) in enumerate(antisense[9:12]) if mo == m)
            feats[f'Aclev_{m}'] = cleav_count
    # GC content
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        gc = sum(1 for b, _ in nts if b in 'GC') / max(len(nts), 1)
        feats[f'{strand_name}_GC'] = gc
    return feats


def parent_key(sense, antisense):
    """Identity of the underlying duplex, ignoring chemical modifications.

    siRNAmod (Martinelli 2023) is built as a small number of PARENT duplexes x
    many modification patterns. Splitting rows at random therefore puts
    different modified versions of the same duplex on both sides of the split,
    and 504 of the 538 features are per-position base/modification one-hots, so
    a tree can identify the parent from the bases alone and recall its mean
    response. Grouping on this key is what makes the CV measure generalisation
    to new duplexes rather than recall of seen ones.
    """
    return (''.join(b for b, _ in sense), ''.join(b for b, _ in antisense))


def load_data():
    rows = []
    fp = os.path.join(DATA, 'martinelli_2023_1.csv.gz')
    with gzip.open(fp, 'rt', encoding='utf-8', errors='replace') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    X_dicts, y_vals, groups = [], [], []
    for r in rows:
        sense, antisense = parse_helm(r['x'])
        X_dicts.append(build_features(sense, antisense))
        y_vals.append(float(r['y']))
        groups.append(parent_key(sense, antisense))
    # 统一特征空间
    all_keys = sorted(set(k for d in X_dicts for k in d))
    X = np.array([[d.get(k, 0) for k in all_keys] for d in X_dicts], dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    g = np.array([hash(t) for t in groups], dtype=np.int64)
    return X, y, all_keys, g


def _params(seed):
    return dict(n_estimators=2000, max_depth=5, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0,
                random_state=seed, n_jobs=-1, verbosity=0,
                early_stopping_rounds=50, eval_metric="rmse")


def evaluate(X, y, groups=None, n_folds=5, seed=42, grouped=True):
    """Nested CV: outer folds score, an inner split picks the tree count.

    Changes from the released configuration, all of which were required for the
    number to mean anything on 907 rows x 538 features (1.7 rows per feature):
      * outer folds are GroupKFold over parent duplexes when `grouped` (the
        default); pass grouped=False to reproduce the old random KFold
      * n_estimators is chosen by early stopping on an inner split of the
        training fold instead of being fixed at 300 by hand
      * the inner split is grouped as well, so tree count is not tuned on data
        that shares a parent duplex with the outer test fold
    """
    from sklearn.model_selection import GroupKFold, KFold, GroupShuffleSplit, train_test_split
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    import xgboost as xgb

    if grouped and groups is not None:
        splitter = GroupKFold(n_splits=n_folds).split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X)

    pearsons, spearmans, rmses, best_iters = [], [], [], []
    for fold, (tr, te) in enumerate(splitter):
        if grouped and groups is not None:
            inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
            i_tr, i_va = next(inner.split(X[tr], y[tr], groups[tr]))
        else:
            i_tr, i_va = train_test_split(np.arange(len(tr)), test_size=0.2,
                                          random_state=seed + fold)
        m = xgb.XGBRegressor(**_params(seed + fold))
        m.fit(X[tr][i_tr], y[tr][i_tr],
              eval_set=[(X[tr][i_va], y[tr][i_va])], verbose=False)
        pred = m.predict(X[te])
        pearsons.append(pearsonr(y[te], pred)[0])
        spearmans.append(spearmanr(y[te], pred)[0])
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
        best_iters.append(int(getattr(m, "best_iteration", -1)))
    n = len(pearsons)
    return {
        'grouped_by_parent_duplex': bool(grouped and groups is not None),
        'n_folds': n,
        'pearson_mean': float(np.mean(pearsons)),
        'pearson_sd': float(np.std(pearsons, ddof=1)) if n > 1 else None,
        'spearman_mean': float(np.mean(spearmans)),
        'spearman_sd': float(np.std(spearmans, ddof=1)) if n > 1 else None,
        'rmse_mean': float(np.mean(rmses)),
        'fold_pearsons': [round(float(p), 3) for p in pearsons],
        'fold_spearmans': [round(float(s), 3) for s in spearmans],
        'best_iterations': best_iters,
    }


def fit_full(X, y, groups=None, seed=42, n_estimators=None):
    """Fit on all rows and return the model, for shipping as an artifact.

    The released `data/drive_weights/sirnamod_xgb_v1.joblib` had no committed
    producer: nothing in the repository wrote a joblib file, so the artifact
    could not be regenerated or audited. This is that producer.
    """
    import xgboost as xgb
    p = _params(seed)
    p.pop("early_stopping_rounds", None)
    p.pop("eval_metric", None)
    p["n_estimators"] = int(n_estimators) if n_estimators else 300
    m = xgb.XGBRegressor(**p)
    m.fit(X, y)
    return m


if __name__ == '__main__':
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', '..', 'results', 'sirnamod_v1'))
    ap.add_argument('--save_model', type=str, default='',
                    help='path to write the full-data joblib artifact')
    ap.add_argument('--also_random_cv', action='store_true',
                    help='additionally report the old ungrouped random KFold, to '
                         'quantify how much of it was parent-duplex recall')
    a = ap.parse_args()

    X, y, feat_names, groups = load_data()
    n_parents = len(set(groups.tolist()))
    print(f'X: {X.shape}, y: {y.shape} | y range: {y.min():.1f} - {y.max():.1f} | '
          f'mean: {y.mean():.1f}')
    print(f'features: {len(feat_names)} | parent duplexes: {n_parents} | '
          f'rows per feature: {X.shape[0] / X.shape[1]:.2f}')

    res = evaluate(X, y, groups, seed=a.seed, grouped=True)
    print('\n=== 5-fold CV, GROUPED by parent duplex (nested early stopping) ===')
    print(f"Pearson:  {res['pearson_mean']:.3f} ± {res['pearson_sd']:.3f}")
    print(f"Spearman: {res['spearman_mean']:.3f} ± {res['spearman_sd']:.3f}")
    print(f"RMSE:     {res['rmse_mean']:.2f}")
    print(f"Folds:    {res['fold_pearsons']}")

    out = {'grouped': res, 'n_parent_duplexes': n_parents,
           'n_rows': int(X.shape[0]), 'n_features': int(X.shape[1])}
    if a.also_random_cv:
        rnd = evaluate(X, y, groups, seed=a.seed, grouped=False)
        out['ungrouped_random_kfold'] = rnd
        print('\n=== 5-fold CV, UNGROUPED random KFold (released configuration) ===')
        print(f"Pearson:  {rnd['pearson_mean']:.3f} ± {rnd['pearson_sd']:.3f}")
        print(f"  (the gap to the grouped result is the part of the released "
              f"number attributable to seeing the same parent duplex in both "
              f"splits, not to predicting modification effects)")

    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, 'cv_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(a.out_dir, 'cv_results.json')}")

    if a.save_model:
        import joblib
        n_est = int(np.median([b for b in res['best_iterations'] if b > 0]) or 300)
        m = fit_full(X, y, groups, seed=a.seed, n_estimators=n_est)
        joblib.dump({'model': m, 'feature_names': feat_names,
                     'n_estimators': n_est, 'seed': a.seed,
                     'cv': res}, a.save_model)
        print(f'wrote {a.save_model} (n_estimators={n_est} from grouped-CV median)')

    # 修饰效应分析：各修饰类型的平均抑制率差异
    print('\n=== 修饰效应 ===')
    for m in ['fl2r', 's4r', 'lna', 'hna', 'una']:
        idx_m = [i for i, fn in enumerate(feat_names) if fn == f'sense_n_{m}']
        if not idx_m:
            continue
        j = idx_m[0]
        has = X[:, j] > 0
        if has.sum() > 5:
            print(f'  sense {m}: n={int(has.sum())} | y={y[has].mean():.1f} vs 无={y[~has].mean():.1f} | Δ={y[has].mean()-y[~has].mean():+.1f}')
