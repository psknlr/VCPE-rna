"""Cheap, strong baselines for the perturbation-response head.

Motivation
----------
Before the v4 audit this repository computed exactly one baseline for the
response line -- "predict no change" -- and quoted every external model (GEARS,
scGPT, CPA, AIDO.RNA-Pert) from its paper without running it. That leaves the
central question unanswered: does a 5.7M-parameter conditioned head do anything
that a retrieval rule over the same frozen ESM2 embeddings does not?

This module supplies the controls that answer it. They are deliberately the
*cheap* ones, because in perturbation-response prediction cheap baselines have
repeatedly proved hard to beat -- and a learned model that does not clear them
has not been shown to be learning perturbation biology at all.

All baselines are fit on training perturbations only and predict the residual
`dev` (full fold-change minus the train-only common core), i.e. exactly the
quantity the model predicts, so their numbers are directly comparable and can
be fed to the same masked metrics.

Baselines
---------
zero
    Predict no residual. Equivalent to "the response is entirely the shared
    core". This is the floor any conditioning must clear.

train_mean
    Predict the mean training residual, ignoring which gene was perturbed. Any
    model scoring at this level has learned no perturbation-specific signal,
    however good its correlation looks.

knn_esm2
    For a test perturbation, average the residual profiles of its k nearest
    TRAINING perturbations in frozen ESM2 space (cosine similarity, optionally
    similarity-weighted). This is the retrieval control: it uses the same
    conditioning information the model receives, with no learning at all. If the
    learned head does not beat it, the head is an expensive nearest-neighbour
    lookup.

neighbor_prior
    The train mean plus ONE parameter for "this gene is a STRING partner of the
    perturbed gene". It exists so that a gain from the model's `is_neighbor`
    channel can be attributed: that channel is information the ESM2 controls
    cannot express, so without a graph-aware control a gain from extra input
    would look like a gain from architecture.

ridge_esm2
    Closed-form multivariate ridge from the target gene's ESM2 embedding to the
    residual profile. A linear map over the same inputs; the standard "is a
    linear baseline enough?" control.

rff_ridge_esm2
    Ridge over random Fourier features of the same ESM2 vectors, i.e. a generic
    NON-LINEAR map, fitted in closed form. It exists because of what the project
    is actually trying to establish. The head beat `ridge_esm2` by +0.027 on the
    plain 150M table and tied it on the graph-augmented one, and the published
    reading of that was "the head's non-linearity was substituting for
    information the representation lacked". That reading has no control: between
    "a linear map on these features" and "this architecture" there is a rung
    missing, and without it any gap over ridge is attributed to the architecture
    by default. This is the rung. If it matches the head, what the head has is
    non-linearity, not architecture; if the head beats it, the architecture is
    doing something a generic non-linear map on the same features cannot.

Each returns a `[n_test, n_hvg]` prediction matrix.

Hyperparameters
---------------
`ridge_esm2` ran at `alpha=1.0` and `knn_esm2` at `k=10` for every number this
repository has published, chosen once and never selected, while the head's epoch
is selected on an inner split and its lr / width / weight decay were tuned over
many runs on this data. A control that was not allowed to fit its one knob is a
weaker control than it looks, and the comparison it loses is not evidence about
architectures.

`select_control_hyperparams` fixes that with the discipline the head already has:
every knob is chosen on the inner-validation slice alone, the full grid is
returned so the choice is auditable, and the held-out set is never consulted. The
defaults stay available and are still reported under the original names, because
silently changing what `ridge_esm2` means between two tables is ERRATA E4's
failure mode.
"""
import numpy as np

from eval_metrics import per_item_correlation


def _l2_normalise(X, eps=1e-8):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def baseline_zero(dev_tr, esm_tr, esm_te, mask_tr=None):
    """Predict no residual at all."""
    return np.zeros((len(esm_te), dev_tr.shape[1]), dtype=np.float32)


def baseline_train_mean(dev_tr, esm_tr, esm_te, mask_tr=None):
    """Predict the (masked) mean training residual for every test perturbation."""
    if mask_tr is None:
        mu = dev_tr.mean(axis=0)
    else:
        denom = mask_tr.sum(axis=0)
        mu = np.where(denom > 0, (dev_tr * mask_tr).sum(axis=0) / np.maximum(denom, 1), 0.0)
    return np.tile(mu.astype(np.float32), (len(esm_te), 1))


def baseline_knn_esm2(dev_tr, esm_tr, esm_te, mask_tr=None, k=10, weighted=True,
                      exclude_self_threshold=0.9999):
    """Average the residuals of the k nearest training perturbations in ESM2 space.

    `exclude_self_threshold` drops training neighbours whose cosine similarity is
    essentially 1.0. Without it, a test perturbation whose target gene also
    appears in training (which is precisely the leak of ERRATA E5) retrieves
    itself and this "baseline" reports memorisation rather than retrieval.
    """
    A = _l2_normalise(np.asarray(esm_te, dtype=np.float64))
    B = _l2_normalise(np.asarray(esm_tr, dtype=np.float64))
    sim = A @ B.T                                     # [n_te, n_tr]
    sim[sim >= exclude_self_threshold] = -np.inf

    kk = min(k, sim.shape[1])
    idx = np.argpartition(-sim, kth=kk - 1, axis=1)[:, :kk]
    out = np.zeros((len(A), dev_tr.shape[1]), dtype=np.float32)
    for i in range(len(A)):
        nb = idx[i]
        s = sim[i, nb]
        ok = np.isfinite(s)
        if not ok.any():
            continue
        nb, s = nb[ok], s[ok]
        if weighted:
            w = np.clip(s, 0.0, None)
            if w.sum() <= 0:
                w = np.ones_like(w)
        else:
            w = np.ones_like(s)
        w = w / w.sum()
        if mask_tr is None:
            out[i] = (dev_tr[nb] * w[:, None]).sum(axis=0)
        else:
            # average only over neighbours that actually measured each gene
            m = mask_tr[nb].astype(np.float64)
            num = (dev_tr[nb] * m * w[:, None]).sum(axis=0)
            den = (m * w[:, None]).sum(axis=0)
            out[i] = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
    return out


def baseline_ridge_esm2(dev_tr, esm_tr, esm_te, mask_tr=None, alpha=1.0,
                        center=True):
    """Closed-form ridge from the ESM2 embedding to the residual profile.

    Solves (X'X + alpha*I) W = X'Y in feature space. With 5120-dimensional ESM2
    vectors and a few thousand training perturbations this is a small solve.
    """
    X = np.asarray(esm_tr, dtype=np.float64)
    Y = np.asarray(dev_tr, dtype=np.float64)
    Xte = np.asarray(esm_te, dtype=np.float64)
    if center:
        x_mu, y_mu = X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True)
        X, Y, Xte = X - x_mu, Y - y_mu, Xte - x_mu
    else:
        y_mu = 0.0

    d = X.shape[1]
    n = X.shape[0]
    if d <= n:
        A = X.T @ X + alpha * np.eye(d)
        W = np.linalg.solve(A, X.T @ Y)
        pred = Xte @ W
    else:
        # dual / kernel form is cheaper when features outnumber samples
        K = X @ X.T + alpha * np.eye(n)
        Alpha = np.linalg.solve(K, Y)
        pred = (Xte @ X.T) @ Alpha
    return (pred + y_mu).astype(np.float32)


def _median_sq_distance(X, rng, n_sample=512):
    """Median squared pairwise distance -- the usual RBF bandwidth heuristic.

    Taken on a subsample so the cost does not grow with the training set, and
    floored away from zero so a degenerate table cannot produce gamma = inf.
    """
    n = len(X)
    idx = rng.choice(n, size=min(n, n_sample), replace=False)
    S = X[idx]
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    iu = np.triu_indices(len(S), k=1)
    med = float(np.median(d2[iu])) if iu[0].size else 1.0
    return max(med, 1e-8)


def rff_features(X, W, b):
    """Random Fourier feature map for the RBF kernel: sqrt(2/D) cos(Wx + b)."""
    return np.sqrt(2.0 / W.shape[1]) * np.cos(np.asarray(X, dtype=np.float64) @ W + b)


def make_rff(esm_dim, n_features=1024, gamma=1.0, seed=0):
    """Draw the feature map. Deterministic given `seed`, so the control is too.

    A control whose score moves when it is re-run cannot be compared against
    anything (cf. ERRATA E14a, where GEARS's non-determinism at a fixed seed made
    a single run meaningless). The draw is therefore seeded explicitly and the
    seed travels in the report.
    """
    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(esm_dim, n_features))
    b = rng.uniform(0.0, 2.0 * np.pi, size=n_features)
    return W, b


def baseline_rff_ridge_esm2(dev_tr, esm_tr, esm_te, mask_tr=None, alpha=1.0,
                            n_features=1024, gamma_mult=1.0, gamma=None, seed=0):
    """Closed-form ridge over random Fourier features of the ESM2 vectors.

    A generic non-linear map on exactly the features `ridge_esm2` and the head
    see, with no gradient steps and no architecture. `gamma` defaults to
    `gamma_mult / median_squared_distance`, the standard median heuristic, so the
    bandwidth is set by the embedding's own scale rather than guessed.
    """
    X = np.asarray(esm_tr, dtype=np.float64)
    Xte = np.asarray(esm_te, dtype=np.float64)
    Y = np.asarray(dev_tr, dtype=np.float64)
    if gamma is None:
        med = _median_sq_distance(X, np.random.default_rng(seed + 101))
        gamma = float(gamma_mult) / med
    W, b = make_rff(X.shape[1], n_features=n_features, gamma=gamma, seed=seed)
    Z, Zte = rff_features(X, W, b), rff_features(Xte, W, b)

    z_mu, y_mu = Z.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True)
    Z, Zte, Yc = Z - z_mu, Zte - z_mu, Y - y_mu
    d, n = Z.shape[1], Z.shape[0]
    if d <= n:
        Wt = np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ Yc)
        pred = Zte @ Wt
    else:
        A = np.linalg.solve(Z @ Z.T + alpha * np.eye(n), Yc)
        pred = (Zte @ Z.T) @ A
    return (pred + y_mu).astype(np.float32)


def select_control_hyperparams(dev_fit, esm_fit, dev_val, esm_val,
                               mask_fit=None, mask_val=None,
                               ridge_alphas=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4),
                               knn_ks=(1, 3, 5, 10, 20, 50),
                               rff_alphas=(1e-1, 1.0, 1e1, 1e2),
                               rff_gamma_mults=(0.25, 1.0, 4.0),
                               rff_n_features=1024, seed=0):
    """Choose each control's hyperparameters on the inner-validation slice only.

    Fit on `*_fit` (the items the head's gradient sees), score on `*_val` (the
    items that select the head's epoch), never touch the held-out set. That is
    the same rule the head is held to, which is the whole point: a control that
    is denied selection is not a fair comparator, and the repository's central
    open question -- whether the architecture earns anything over a linear map on
    the same features -- is decided by exactly this comparison.

    Returns the chosen values and the full grids, so a reader can see whether the
    choice sat on a plateau or on a spike, and whether it hit the edge of the
    grid (which would mean the grid, not the data, picked it).
    """
    out = {"selected_on": "inner_val", "estimator": "per_item_correlation",
           "n_fit_items": int(len(esm_fit)), "n_val_items": int(len(esm_val))}

    def _score(pred):
        v = per_item_correlation(dev_val, pred, mask_val)
        return None if not np.isfinite(v) else float(v)

    grids = {}
    for name, values, fn in (
        ("ridge_alpha", list(ridge_alphas),
         lambda a: baseline_ridge_esm2(dev_fit, esm_fit, esm_val,
                                       mask_tr=mask_fit, alpha=a)),
        ("knn_k", list(knn_ks),
         lambda k: baseline_knn_esm2(dev_fit, esm_fit, esm_val,
                                     mask_tr=mask_fit, k=int(k))),
    ):
        scored = [(v, _score(fn(v))) for v in values]
        grids[name] = [{"value": v, "inner_val": s} for v, s in scored]
        ok = [(v, s) for v, s in scored if s is not None]
        if ok:
            best = max(ok, key=lambda t: t[1])
            out[name] = best[0]
            out[name + "_inner_val"] = best[1]
            out[name + "_at_grid_edge"] = bool(best[0] in (values[0], values[-1]))

    rff_scored = []
    for a in rff_alphas:
        for g in rff_gamma_mults:
            s = _score(baseline_rff_ridge_esm2(
                dev_fit, esm_fit, esm_val, mask_tr=mask_fit, alpha=a,
                n_features=rff_n_features, gamma_mult=g, seed=seed))
            rff_scored.append({"alpha": a, "gamma_mult": g, "inner_val": s})
    grids["rff"] = rff_scored
    ok = [r for r in rff_scored if r["inner_val"] is not None]
    if ok:
        best = max(ok, key=lambda r: r["inner_val"])
        out["rff_alpha"] = best["alpha"]
        out["rff_gamma_mult"] = best["gamma_mult"]
        out["rff_inner_val"] = best["inner_val"]
        out["rff_n_features"] = int(rff_n_features)
        out["rff_seed"] = int(seed)
    out["grids"] = grids
    return out


def baseline_neighbor_prior(dev_tr, esm_tr, esm_te, mask_tr=None,
                            is_nb_tr=None, is_nb_te=None):
    """Graph-only control: the train mean, plus one parameter for "is a STRING partner".

    This exists to keep a fairness question answerable. The `is_neighbor` channel
    gives the head a biological prior -- knocking a gene down should perturb its
    interaction partners -- that `ridge_esm2` and `knn_esm2` structurally cannot
    express, because they see only the target's ESM2 vector. A head that gains
    from that channel has therefore been given more information, and attributing
    the gain to its architecture instead would be the same mistake this
    repository's errata are about.

    So this predicts the weakest thing that still uses the graph:

        dev_hat[i, h] = mu[h] + beta * is_neighbor[i, h]

    `mu` is the per-gene masked training mean -- i.e. exactly `train_mean` -- and
    `beta` is a single scalar fitted on training items only: how much more a gene
    deviates when it is a partner of the perturbed gene than when it is not. The
    gap between this and `train_mean` is what the graph buys on its own; the gap
    between the head and this is what the head's use of the graph buys beyond a
    one-parameter rule.

    With no graph supplied it degrades to `train_mean`, which is the honest
    fallback rather than an error: the caller may simply not have a table.
    """
    dev = np.asarray(dev_tr, dtype=np.float64)
    m = (np.ones_like(dev, dtype=bool) if mask_tr is None
         else np.asarray(mask_tr, dtype=bool))
    denom = m.sum(axis=0)
    mu = np.where(denom > 0, (dev * m).sum(axis=0) / np.maximum(denom, 1), 0.0)

    n_te = np.asarray(esm_te).shape[0]
    if is_nb_tr is None or is_nb_te is None:
        return np.tile(mu.astype(np.float32), (n_te, 1))

    nb_tr = np.asarray(is_nb_tr, dtype=bool)
    resid = dev - mu[None, :]
    # beta: mean residual on partner positions minus on non-partner positions,
    # both masked. Fitted on TRAINING items only.
    on, off = m & nb_tr, m & ~nb_tr
    n_on, n_off = on.sum(), off.sum()
    if n_on == 0 or n_off == 0:
        beta = 0.0
    else:
        beta = float((resid * on).sum() / n_on - (resid * off).sum() / n_off)
    pred = mu[None, :] + beta * np.asarray(is_nb_te, dtype=np.float64)
    return pred.astype(np.float32)


REGISTRY = {
    "zero": baseline_zero,
    "train_mean": baseline_train_mean,
    "knn_esm2": baseline_knn_esm2,
    "ridge_esm2": baseline_ridge_esm2,
    "neighbor_prior": baseline_neighbor_prior,
    # The `_sel` rows are the same estimators with their one knob chosen on the
    # inner-validation slice. They are separate names rather than new defaults so
    # that `ridge_esm2` means in this table what it meant in the last one.
    "ridge_esm2_sel": baseline_ridge_esm2,
    "knn_esm2_sel": baseline_knn_esm2,
    "rff_ridge_esm2_sel": baseline_rff_ridge_esm2,
}

# Which default each un-selected control runs at. Quoting a control without its
# hyperparameter is the same error as quoting a score without its estimator.
DEFAULTS = {"ridge_esm2": {"alpha": 1.0}, "knn_esm2": {"k": 10}}


def run_all(dev_tr, esm_tr, esm_te, mask_tr=None, which=None, selected=None, **kw):
    """Return {name: prediction matrix} for the requested baselines.

    `selected` is the output of `select_control_hyperparams`. The `*_sel` rows
    need it and are skipped without it, rather than silently falling back to the
    defaults and reporting them twice under two names.
    """
    names = which or [n for n in REGISTRY if not n.endswith("_sel")]
    sel = selected or {}
    out = {}
    for n in names:
        fn = REGISTRY[n]
        if n == "knn_esm2":
            out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr,
                        k=kw.get("knn_k", 10))
        elif n == "ridge_esm2":
            out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr,
                        alpha=kw.get("ridge_alpha", 1.0))
        elif n == "neighbor_prior":
            out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr,
                        is_nb_tr=kw.get("is_nb_tr"), is_nb_te=kw.get("is_nb_te"))
        elif n == "ridge_esm2_sel":
            if "ridge_alpha" in sel:
                out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr,
                            alpha=sel["ridge_alpha"])
        elif n == "knn_esm2_sel":
            if "knn_k" in sel:
                out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr,
                            k=int(sel["knn_k"]))
        elif n == "rff_ridge_esm2_sel":
            if "rff_alpha" in sel:
                out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr,
                            alpha=sel["rff_alpha"],
                            gamma_mult=sel["rff_gamma_mult"],
                            n_features=sel.get("rff_n_features", 1024),
                            seed=sel.get("rff_seed", 0))
        else:
            out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr)
    return out
