"""Evaluation utilities for wer."""

from rapidfuzz.distance import Levenshtein

from src.evaluation.text_normalizer import TextNormalizer


def word_error_breakdown(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Return WER and operation-level counts (S/I/D) using rapidfuzz editops.

    Uses a single alignment pass via rapidfuzz so that the WER value and the
    S/I/D counts are guaranteed to be consistent with each other.
    """
    ref_words = TextNormalizer.normalize(reference).split()
    hyp_words = TextNormalizer.normalize(hypothesis).split()

    n = len(ref_words)

    editops = Levenshtein.editops(ref_words, hyp_words)
    substitutions = sum(1 for op in editops if op.tag == "replace")
    deletions = sum(1 for op in editops if op.tag == "delete")
    insertions = sum(1 for op in editops if op.tag == "insert")

    denominator = max(1, n)
    wer = (substitutions + insertions + deletions) / denominator
    return {
        "wer": wer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "reference_word_count": n,
        "substitution_rate": substitutions / denominator,
        "insertion_rate": insertions / denominator,
        "deletion_rate": deletions / denominator,
    }


def word_error_rate(reference: str, hypothesis: str) -> float:
    return word_error_breakdown(reference, hypothesis)["wer"]
