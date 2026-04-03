"""Evaluation utilities for text normalizer."""

import re
import unicodedata


class TextNormalizer:
    """Normalize clinical text before WER comparison."""

    # Apostrophe-like characters removed without extra spaces (contractions → single tokens).
    _APOSTROPHE_PATTERN = re.compile(r"['\u2019\u2018`´]+")
    _PUNCT_PATTERN = re.compile(r"[^\w\s]")
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    @classmethod
    def normalize(cls, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = normalized.lower()
        normalized = cls._APOSTROPHE_PATTERN.sub("", normalized)
        normalized = cls._PUNCT_PATTERN.sub(" ", normalized)
        normalized = cls._WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized
