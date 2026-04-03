"""Tests for wer behavior."""

from src.evaluation.wer import word_error_breakdown, word_error_rate


def test_word_error_rate_identical() -> None:
    assert word_error_rate("patient is stable", "patient is stable") == 0.0


def test_word_error_rate_nonzero() -> None:
    assert word_error_rate("patient is stable", "patient stable") > 0.0


def test_word_error_rate_normalizes_case_and_punctuation() -> None:
    reference = "Patient is stable, today."
    hypothesis = "patient is stable today"
    assert word_error_rate(reference, hypothesis) == 0.0


def test_word_error_rate_normalizes_unicode_apostrophe() -> None:
    reference = "patient's mood improved"
    hypothesis = "patient’s mood improved"
    assert word_error_rate(reference, hypothesis) == 0.0


def test_word_error_breakdown_counts_operations() -> None:
    # ref: a b c ; hyp: a x c d => S=1, I=1, D=0
    breakdown = word_error_breakdown("a b c", "a x c d")
    assert breakdown["substitutions"] == 1
    assert breakdown["insertions"] == 1
    assert breakdown["deletions"] == 0
    assert breakdown["reference_word_count"] == 3
