"""Integration: normalize → WER → word alignment without DB or mocks."""

from __future__ import annotations

from src.evaluation.text_normalizer import TextNormalizer
from src.evaluation.wer import word_error_breakdown
from src.evaluation.word_alignment import align_words, count_ops


def test_normalizer_wer_alignment_chain() -> None:
    ref = "The patient has mild hypertension."
    hyp = "The patient has mile hypertension."
    br = word_error_breakdown(reference=ref, hypothesis=hyp)
    assert br["wer"] > 0
    steps = align_words(ref, hyp)
    ops = count_ops(steps)
    assert ops["matches"] >= 0
    assert TextNormalizer.normalize(ref) != ""
