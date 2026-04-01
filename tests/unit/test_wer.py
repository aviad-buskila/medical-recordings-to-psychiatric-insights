from src.evaluation.wer import word_error_rate


def test_word_error_rate_identical() -> None:
    assert word_error_rate("patient is stable", "patient is stable") == 0.0


def test_word_error_rate_nonzero() -> None:
    assert word_error_rate("patient is stable", "patient stable") > 0.0
