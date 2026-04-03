"""Tests for text normalizer behavior."""

from src.evaluation.text_normalizer import TextNormalizer


def test_normalize_lowercase_and_punctuation() -> None:
    assert TextNormalizer.normalize("Hello, World!") == "hello world"


def test_normalize_strips_apostrophe_in_contraction() -> None:
    assert TextNormalizer.normalize("Don't panic") == "dont panic"


def test_normalize_empty() -> None:
    assert TextNormalizer.normalize("") == ""
