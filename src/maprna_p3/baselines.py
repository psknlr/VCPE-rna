"""Cheap, strong baselines for the perturbation-response head.

Motivation
----------
Before the v4 audit this repository computed exactly one baseline for the
response line -- "predict no change" -- and quoted every external model (GEARS,
scGPT, CPA, AIDO.RNA-Pert) from its paper without running it. That leaves the
central question unanswered: does a ~1M-parameter conditioned head do anything
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

Each returns a `[n_test, n_hvg]` prediction matrix.
"""
import numpy as np


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
}


def run_all(dev_tr, esm_tr, esm_te, mask_tr=None, which=None, **kw):
    """Return {name: prediction matrix} for the requested baselines."""
    names = which or list(REGISTRY)
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
        else:
            out[n] = fn(dev_tr, esm_tr, esm_te, mask_tr=mask_tr)
    return out
