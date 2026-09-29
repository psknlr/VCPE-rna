# -*- coding: utf-8 -*-
"""siRNAmod modification-aware prediction (platform integration entry point).

Usage (1) (efficacy routing): predict_modified_efficiency(sense, antisense,
    mod_pattern) -> corrected efficiency in 0-1
Usage (2) (standalone tool): predict_inhibition(sense, antisense, sense_mods,
    antisense_mods) -> inhibition rate in 0-100
"""
import joblib
import os
import re

import numpy as np

_MODEL = None
_FEAT_NAMES = None
_MOD_ORDER = ['unmod', 'fl2r', 's4r', 'lna', 'hna', 'una', 'ome', 'dna']
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "data", "drive_weights", "sirnamod_xgb_v1.joblib")

# Presets for common modification patterns -> (sense_mods_fn, antisense_mods_fn)
# NOTE: each value is in fact ONE function of `seq` that returns the tuple
# (sense_mods, antisense_mods), not a pair of functions.
MOD_PATTERNS = {
    "unmodified": lambda seq: (['unmod'] * len(seq), ['unmod'] * len(seq)),
    "2F_alternating": lambda seq: (['fl2r' if i % 2 == 0 else 'unmod' for i in range(len(seq))],
                                    ['unmod'] * len(seq)),
    "2OMe_alternating": lambda seq: (['ome' if i % 2 == 0 else 'unmod' for i in range(len(seq))],
                                      ['unmod'] * len(seq)),
    "2F2OMe_standard": lambda seq: (['fl2r' if i % 2 == 0 else 'ome' for i in range(len(seq))],
                                     ['fl2r' if i % 2 == 0 else 'ome' for i in range(len(seq))]),
    "full_2F": lambda seq: (['fl2r'] * len(seq), ['fl2r'] * len(seq)),
    "LNA_alternating": lambda seq: (['lna' if i % 2 == 0 else 'unmod' for i in range(len(seq))],
                                     ['unmod'] * len(seq)),
    "s4r_sense": lambda seq: (['s4r'] * len(seq), ['unmod'] * len(seq)),
}


def _ensure_model():
    global _MODEL, _FEAT_NAMES
    if _MODEL is not None:
        return
    data = joblib.load(_MODEL_PATH)
    _MODEL = data['model']
    # The shipped v1 artifact stores this as 'feat_names'; the producer added in
    # sirnamod_model.py writes 'feature_names'. Accept either, and fail with the
    # keys actually present rather than a bare KeyError, so a mismatch is
    # diagnosable instead of mysterious.
    for key in ('feat_names', 'feature_names'):
        if key in data:
            _FEAT_NAMES = data[key]
            break
    else:
        raise KeyError(
            f"{_MODEL_PATH} has no feature-name list; expected 'feat_names' or "
            f"'feature_names', found {sorted(data)}")

    # The artifact records the modification order its features were built with,
    # but this module previously ignored it and used the hard-coded _MOD_ORDER
    # above. An artifact trained with a different order would then have its
    # features assembled in the wrong order and return confidently wrong
    # predictions, with nothing to indicate it. Validate instead.
    stored = data.get('mod_order')
    if stored is not None and list(stored) != list(_MOD_ORDER):
        raise ValueError(
            f"{_MODEL_PATH} was built with modification order {list(stored)}, but "
            f"this module builds features in the order {list(_MOD_ORDER)}. "
            f"Features would be assembled in the wrong order. Update _MOD_ORDER "
            f"to match the artifact, or retrain the artifact.")


def build_features(sense, antisense, max_len=25):
    """Feature construction exactly identical to sirnamod_model.build_features.

    Args:
        sense: list of (base, mod) tuples
        antisense: list of (base, mod) tuples
    Returns:
        dict of features
    """
    feats = {}
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        mod_counts = {}
        for _, m in nts:
            mod_counts[m] = mod_counts.get(m, 0) + 1
        for m in _MOD_ORDER:
            feats[f'{strand_name}_n_{m}'] = mod_counts.get(m, 0)
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for m in _MOD_ORDER:
                feats[f'{strand_name}{pos}_{m}'] = 1 if mod == m else 0
    for strand_name, nts in (('S', sense), ('A', antisense)):
        for pos, (base, mod) in enumerate(nts[:max_len]):
            for b in 'ACGU':
                feats[f'{strand_name}{pos}_{b}'] = 1 if base == b else 0
    if len(antisense) >= 12:
        for m in _MOD_ORDER:
            seed_count = sum(1 for pos, (b, mo) in enumerate(antisense[1:8]) if mo == m)
            feats[f'Aseed_{m}'] = seed_count
            cleav_count = sum(1 for pos, (b, mo) in enumerate(antisense[9:12]) if mo == m)
            feats[f'Aclev_{m}'] = cleav_count
    for strand_name, nts in (('sense', sense), ('antisense', antisense)):
        gc = sum(1 for b, _ in nts if b in 'GC') / max(len(nts), 1)
        feats[f'{strand_name}_GC'] = gc
    return feats


def predict_inhibition(sense_seq, antisense_seq, sense_mods=None, antisense_mods=None):
    """Predict the inhibition rate % of a modified siRNA.

    Args:
        sense_seq: str guide-strand sequence (19-25 nt)
        antisense_seq: str antisense-strand sequence
        sense_mods: list[str] per-position modification type (None = all unmod)
        antisense_mods: list[str] same as above

    Returns:
        float, predicted inhibition rate % (0-100, clipped to [0, 100])

    NOTE: `sense_seq` is labelled the "guide strand" above, but in an siRNA
    duplex the guide strand is the ANTISENSE strand; the code feeds `sense_seq`
    into the sense-strand features of build_features, so the label, not the
    code, is the inconsistent part. The stated 19-25 nt range is not checked
    anywhere -- build_features only truncates positional features at
    max_len=25.
    """
    _ensure_model()
    sense_nts = [(b, (sense_mods[i] if sense_mods and i < len(sense_mods) else 'unmod'))
                 for i, b in enumerate(sense_seq.upper().replace('T', 'U'))]
    antisense_nts = [(b, (antisense_mods[i] if antisense_mods and i < len(antisense_mods) else 'unmod'))
                     for i, b in enumerate(antisense_seq.upper().replace('T', 'U'))]
    feats = build_features(sense_nts, antisense_nts)
    x = np.array([[feats.get(k, 0) for k in _FEAT_NAMES]], dtype=np.float32)
    pred = float(_MODEL.predict(x)[0])
    return max(0.0, min(100.0, pred))


def predict_modified_efficiency(sense_seq, antisense_seq, mod_pattern="unmodified"):
    """Efficacy-routing integration entry point: modification pattern ->
    corrected knockdown efficiency (0-1).

    Args:
        sense_seq: str guide-strand sequence
        antisense_seq: str antisense-strand sequence
        mod_pattern: str modification-pattern name (a key of MOD_PATTERNS) or None

    Returns:
        (efficiency_0_1, sirnamod_inhibition_pct, mod_label)
        - efficiency: 0-1, sirnamod predicted inhibition rate / 100
        - sirnamod_inhibition_pct: raw predicted inhibition rate %
        - mod_label: modification-pattern description

    NOTE: "corrected" in the summary line corresponds to no correction step in
    the code -- the efficiency returned is exactly the predicted inhibition
    rate divided by 100. `mod_label` is likewise the pattern NAME (`mod_pattern`
    itself, or "unmodified"), not a description of the pattern.
    """
    if mod_pattern and mod_pattern in MOD_PATTERNS:
        sense_mods, antisense_mods = MOD_PATTERNS[mod_pattern](sense_seq)
        label = mod_pattern
    else:
        sense_mods = antisense_mods = None
        label = "unmodified"
    inh = predict_inhibition(sense_seq, antisense_seq, sense_mods, antisense_mods)
    eff = inh / 100.0
    return eff, inh, label


def get_mod_pattern_names():
    """Return the list of supported modification-pattern names."""
    return list(MOD_PATTERNS.keys())
