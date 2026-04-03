"""Tests for mer wil behavior."""

from src.evaluation.mer_wil import word_mer_wil_breakdown


def test_mer_wil_identical() -> None:
    b = word_mer_wil_breakdown("a b c", "a b c")
    assert b["mer"] == 0.0
    assert b["wil"] == 0.0
    assert b["reference_word_count"] == 3
    assert b["hypothesis_word_count"] == 3


def test_mer_wil_formula_matches_edit_over_r_plus_h() -> None:
    # ref: a b ; hyp: a x  => S=1, R=2, H=2, E=1, MER=1/4, WIL=2/4
    b = word_mer_wil_breakdown("a b", "a x")
    assert b["substitutions"] + b["insertions"] + b["deletions"] == 1
    assert b["mer"] == 0.25
    assert b["wil"] == 0.5
