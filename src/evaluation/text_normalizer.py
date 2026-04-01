import re
import unicodedata


class TextNormalizer:
    """Normalize clinical text before WER comparison."""

    _PUNCT_PATTERN = re.compile(r"[^\w\s']")
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    @classmethod
    def normalize(cls, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = normalized.lower()
        normalized = normalized.replace("’", "'")
        normalized = cls._PUNCT_PATTERN.sub(" ", normalized)
        normalized = cls._WHITESPACE_PATTERN.sub(" ", normalized).strip()
        return normalized
