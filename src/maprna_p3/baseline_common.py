"""Shared machinery for comparing an external model against the VCPE head.

Every external baseline has to be put on the same footing, and the alignment work
is identical regardless of which model is being compared: the same split, the same
gene panel, the same predicted quantity, the same mask, the same estimators. This
module holds that once.

It exists as a module rather than as copy-paste for a specific reason. The two
worst defects in docs/ERRATA.md (E1 and E2) both arose because each script
re-implemented its own metrics, so the definitions drifted and one of them ended
up correlating a feature instead of a label. Re-implementing the *comparison* per
baseline would reproduce that failure one level up: two baselines would end up
scored slightly differently and the resulting table would be meaningless.

The conversion that matters
---------------------------
VCPE predicts a residual: `dev = fc - common_fc`, where `fc` is the log
fold-change against the dataset's control mean and `common_fc` is the *train-only*
mean fold-change. Most external models predict post-perturbation expression
instead. Scoring their raw expression against VCPE's residual would flatter them
enormously, because the shared response dominates raw expression -- which is the
entire finding of the P2 erratum. `expression_to_dev` applies the same conversion
so the two are comparable.
"""
import numpy as np

from eval_metrics import (bootstrap_ci_per_item, per_item_correlation,
                          pooled_correlation, top_k_overlap)


def hvg_symbols(data):
    """VCPE HVG row ids -> gene symbols, for aligning an external gene space."""
    row2sym = {r: s for s, r in data["sym2row"].items()}
    return [row2sym.get(int(r)) for r in np.asarray(data["hvg_rows"])]


def inner_split_items(data):
    """(train, val, test) item lists, matching what train_p3 trains and scores on."""
    is_val = np.asarray(data["is_inner_val"])
    tr = [it for it, v in zip(data["train_items"], is_val) if not v]
    va = [it for it, v in zip(data["train_items"], is_val) if v]
    return tr, va, data["test_items"]


def align_to_panel(values, source_gene_names, panel_symbols):
    """Project a vector over `source_gene_names` onto `panel_symbols` by SYMBOL.

    Returns (vector, found_mask). Positional alignment would be silently wrong
    whenever the two gene orders differ, which they generally do.

    Genes the external model does not cover are left at zero and marked absent.
    That distinction matters: counting an uncovered gene as 0.0 would credit the
    model with correctly predicting no change on a gene it was never asked about.
    """
    idx = {str(g): i for i, g in enumerate(source_gene_names)}
    out = np.zeros(len(panel_symbols), dtype=np.float32)
    found = np.zeros(len(panel_symbols), dtype=bool)
    for j, sym in enumerate(panel_symbols):
        if sym is None:
            continue
        i = idx.get(str(sym))
        if i is not None:
            out[j] = values[i]
            found[j] = True
    return out, found


def expression_to_dev(pred_expr, ctrl_expr, common_fc):
    """Convert predicted EXPRESSION to the residual VCPE predicts.

    dev = (pred_expr - ctrl_expr) - common_fc

    `ctrl_expr` must be the same control mean and `common_fc` the same train-only
    common core that VCPE's target was built from, or the two models are not
    predicting the same quantity.
    """
    return (np.asarray(pred_expr) - np.asarray(ctrl_expr)) - np.asarray(common_fc)


def score(true_dev, pred_dev, mask, seed=0, n_boot=500, k=50):
    """The one scoring function every baseline goes through."""
    return dict(
        pearson_dev=per_item_correlation(true_dev, pred_dev, mask),
        pearson_dev_pooled=pooled_correlation(true_dev, pred_dev, mask),
        top50_dev=top_k_overlap(true_dev, pred_dev, k=k, mask=mask),
        pearson_dev_ci95=list(bootstrap_ci_per_item(true_dev, pred_dev, mask,
                                                    n_boot=n_boot, seed=seed)),
        pearson_dev_estimator="mean_over_perturbations_of_within_perturbation_r",
    )


def intersect_masks(vcpe_mask, coverage_mask):
    """Score only where BOTH a ground truth and a prediction exist.

    A column VCPE never measured has no ground truth; a column the external model
    does not cover has no prediction. Scoring either as "no change" would credit
    or penalise a model for a gene that was never in play.
    """
    return np.asarray(vcpe_mask) & np.asarray(coverage_mask)


#: Caveats that apply to every external comparison here and must travel with any
#: number produced from one. Stated as data so each adapter reports them
#: identically rather than paraphrasing.
UNCONTROLLED_FACTORS = [
    "Tuning effort is not equalised: the external model runs at or near its "
    "defaults, while VCPE's hyperparameters were chosen against this data over "
    "many runs. That asymmetry favours VCPE and no protocol alignment removes it; "
    "the honest fix is an equal tuning budget for both.",
    "Capacity is not equalised: the external models are generally far larger than "
    "the 5.7M-parameter VCPE head. If VCPE wins it wins as the smaller model; if "
    "it loses, the size difference is an explanation rather than an excuse.",
    "Each external model keeps its own preprocessing (its own graph, its own "
    "highly-variable-gene or DE selection). Disabling that would not be a "
    "comparison against the published method.",
]


def fairness_block(extra=()):
    """Standard fairness section for a baseline report."""
    return dict(
        controlled=[
            "Split: the external model is given VCPE's exact train/val/test "
            "condition sets, so both hold out the same perturbations (the same "
            "held-out genes under --split_by target_gene).",
            "Predicted quantity: expression predictions are converted to the same "
            "residual VCPE predicts (fold-change against the same control mean, "
            "minus the same train-only common core). Scoring raw expression "
            "against a residual would flatter the external model, since the "
            "shared response dominates raw expression.",
            "Gene space and mask: VCPE's HVG panel, with VCPE's measured mask "
            "intersected with the external model's own gene coverage.",
            "Estimator: the same per-item and pooled correlation functions, named "
            "explicitly so the two are never mixed.",
        ],
        not_controlled=list(UNCONTROLLED_FACTORS) + list(extra),
    )
