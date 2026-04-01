from src.evaluation.wer import word_error_rate


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
