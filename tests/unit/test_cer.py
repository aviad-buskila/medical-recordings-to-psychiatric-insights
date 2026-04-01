from src.evaluation.cer import align_char_lists_with_indices, character_error_breakdown


def test_cer_identical() -> None:
    b = character_error_breakdown("hello", "hello")
    assert b["cer"] == 0.0
    assert b["substitutions"] == b["insertions"] == b["deletions"] == 0


def test_cer_one_substitution() -> None:
    b = character_error_breakdown("ab", "ax")
    assert b["reference_character_count"] == 2
    assert b["substitutions"] == 1


def test_align_char_indices_consistent_with_counts() -> None:
    ref = list("kitten")
    hyp = list("sitting")
    b = character_error_breakdown("kitten", "sitting")
    steps = align_char_lists_with_indices(ref, hyp)
    assert sum(1 for op, _, _ in steps if op == "S") == b["substitutions"]
    assert sum(1 for op, _, _ in steps if op == "I") == b["insertions"]
    assert sum(1 for op, _, _ in steps if op == "D") == b["deletions"]
