"""Tests for near-duplicate detection and split-leakage measurement.

The relation being tested is "shares a long exact substring", chosen because the
two processes that generate duplicates in ASO patent corpora -- family
republication and single-nucleotide target walking -- both produce sequences that
share a long substring while differing as whole strings. Whole-string edit
distance does not separate those from genuinely different oligonucleotides: a
1-nt shift has edit distance 2 but is the same binding site.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "efficacy"))

from seq_dedup import analyse, find_near_duplicates, is_near_duplicate  # noqa: E402


# --------------------------------------------------------------------------
# the pairwise relation
# --------------------------------------------------------------------------

def test_identical_sequences_are_near_duplicates():
    s = "ACGTACGTACGTACGT"
    assert is_near_duplicate(s, s, 0.9)


def test_single_nucleotide_walk_is_caught():
    """The tiling ladder: the shared substring is 15 of 16 positions."""
    a = "ACGTTGCAACGTTGCA"
    b = "CGTTGCAACGTTGCAT"          # same site, shifted by one
    assert is_near_duplicate(a, b, 0.9)


def test_trim_at_one_end_is_caught():
    a = "ACGTTGCAACGTTGCA"
    b = a[:14]                       # a re-file two nucleotides shorter
    assert is_near_duplicate(a, b, 0.9)


def test_unrelated_sequences_are_not_near_duplicates():
    a = "ACGTACGTACGTACGT"
    b = "TTTTGGGGCCCCAAAA"
    assert not is_near_duplicate(a, b, 0.9)


def test_a_mid_sequence_substitution_breaks_the_relation_at_high_threshold():
    """A single internal mismatch halves the longest shared substring.

    This is deliberate: an internal substitution is a different molecule with
    potentially different potency, whereas a shift is the same site. The metric
    should separate them, and at 0.9 it does.
    """
    a = "ACGTTGCAACGTTGCA"
    b = "ACGTTGCTACGTTGCA"          # one internal change, position 8
    assert not is_near_duplicate(a, b, 0.9)
    assert is_near_duplicate(a, b, 0.5)


def test_threshold_is_relative_to_the_shorter_sequence():
    short = "ACGTACGTAC"                       # 10 nt
    long = "TTTT" + short + "TTTT"             # fully contains it
    assert is_near_duplicate(short, long, 1.0)


# --------------------------------------------------------------------------
# the indexed search
# --------------------------------------------------------------------------

def test_index_finds_a_tiling_ladder():
    """Six sequences walking a target one nucleotide at a time."""
    target = "ACGTTGCAACGTTGCAACGTTGCAACGT"
    ladder = [target[i:i + 16] for i in range(6)]
    pairs = find_near_duplicates(ladder, threshold=0.9, k=12)
    # every adjacent rung must be found; most non-adjacent ones too
    adjacent = {(i, i + 1) for i in range(5)}
    assert adjacent <= set(pairs)


def test_index_returns_nothing_for_unrelated_sequences():
    seqs = ["ACGTACGTACGTACGT", "TTTTGGGGCCCCAAAA", "GGGGTTTTAAAACCCC"]
    assert find_near_duplicates(seqs, threshold=0.9, k=12) == []


def test_pairs_are_reported_once_and_ordered():
    seqs = ["ACGTTGCAACGTTGCA", "CGTTGCAACGTTGCAT"]
    pairs = find_near_duplicates(seqs, threshold=0.9, k=12)
    assert pairs == [(0, 1)]


# --------------------------------------------------------------------------
# split-leakage measurement -- the point of the script
# --------------------------------------------------------------------------

def _ladder_rows(group_of):
    """A 6-rung tiling ladder on one target, assigned to groups by `group_of`."""
    target = "ACGTTGCAACGTTGCAACGTTGCAACGT"
    return [{"seq": target[i:i + 16], "target": "GENE1", "grp": group_of(i)}
            for i in range(6)]


def test_a_grouping_that_splits_a_ladder_is_flagged_as_leaking():
    rep = analyse(_ladder_rows(lambda i: f"patent_table_{i}"),
                  threshold=0.9, k=12, group_cols=["grp"])
    g = rep["by_group_column"]["grp"]
    assert rep["n_near_duplicate_pairs"] > 0
    assert g["near_duplicate_pairs_crossing_groups"] == rep["n_near_duplicate_pairs"]
    assert g["fraction_of_near_duplicate_pairs_crossing"] == 1.0
    assert "does NOT control" in g["verdict"]


def test_a_grouping_that_keeps_a_ladder_together_is_not_flagged():
    rep = analyse(_ladder_rows(lambda i: "one_patent"),
                  threshold=0.9, k=12, group_cols=["grp"])
    g = rep["by_group_column"]["grp"]
    assert g["near_duplicate_pairs_crossing_groups"] == 0
    assert "controls" in g["verdict"] and "NOT" not in g["verdict"]


def test_exact_duplicates_spanning_groups_are_counted_separately():
    """Family republication: the same string under two different tables."""
    s = "ACGTTGCAACGTTGCA"
    rows = [{"seq": s, "target": "GENE1", "grp": "table_A"},
            {"seq": s, "target": "GENE1", "grp": "table_B"}]
    rep = analyse(rows, threshold=0.9, k=12, group_cols=["grp"])
    assert rep["n_exact_duplicate_groups"] == 1
    assert rep["largest_exact_duplicate_group"] == 2
    assert rep["by_group_column"]["grp"]["exact_duplicate_groups_spanning_groups"] == 1


def test_comparisons_are_confined_to_a_shared_target_by_default():
    """A shared substring across different targets is biologically unremarkable."""
    s = "ACGTTGCAACGTTGCA"
    rows = [{"seq": s, "target": "GENE1", "grp": "a"},
            {"seq": s, "target": "GENE2", "grp": "b"}]
    confined = analyse(rows, threshold=0.9, k=12, group_cols=["grp"])
    pooled = analyse(rows, threshold=0.9, k=12, group_cols=["grp"], by_target=False)
    assert confined["n_near_duplicate_pairs"] == 0
    assert pooled["n_near_duplicate_pairs"] == 1
    # exact-duplicate accounting is target-independent, so it sees both either way
    assert confined["n_exact_duplicate_groups"] == 1


def test_report_counts_unique_sequences():
    s = "ACGTTGCAACGTTGCA"
    rows = [{"seq": s, "target": "G", "grp": "a"}] * 3 + \
           [{"seq": "TTTTGGGGCCCCAAAA", "target": "G", "grp": "a"}]
    rep = analyse(rows, threshold=0.9, k=12, group_cols=["grp"])
    assert rep["n_rows"] == 4
    assert rep["n_unique_sequences"] == 2
    assert rep["n_rows_in_exact_duplicate_groups"] == 3
