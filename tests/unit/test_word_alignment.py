import random

from src.evaluation.wer import word_error_breakdown
from src.evaluation.word_alignment import align_words, count_ops


def test_alignment_counts_match_wer_breakdown() -> None:
    random.seed(42)
    vocab = ["the", "a", "patient", "had", "pain", "x", "y"]
    for _ in range(80):
        ref = " ".join(random.choices(vocab, k=random.randint(0, 12)))
        hyp = " ".join(random.choices(vocab, k=random.randint(0, 12)))
        steps = align_words(ref, hyp)
        c = count_ops(steps)
        br = word_error_breakdown(ref, hyp)
        assert c["substitutions"] == br["substitutions"]
        assert c["insertions"] == br["insertions"]
        assert c["deletions"] == br["deletions"]


def test_alignment_simple_substitution() -> None:
    steps = align_words("hello world", "hello wrld")
    ops = count_ops(steps)
    assert ops["substitutions"] == 1
    assert ops["matches"] == 1


def test_alignment_insertion() -> None:
    steps = align_words("a b", "a x b")
    ops = count_ops(steps)
    assert ops["insertions"] == 1
    assert ops["matches"] == 2
