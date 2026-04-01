from src.evaluation.speaker_wer import per_speaker_breakdown, speaker_labels_for_reference_words


def test_per_speaker_substitution_on_speaker_two() -> None:
    ref_words = ["a", "b"]
    hyp_words = ["a", "x"]
    speaker_per_ref = [1, 2]
    out = per_speaker_breakdown(ref_words, hyp_words, speaker_per_ref)
    assert out[1]["substitutions"] == 0 and out[1]["deletions"] == 0 and out[1]["insertions"] == 0
    assert out[2]["substitutions"] == 1
    assert out[1]["reference_word_count"] == 1
    assert out[2]["reference_word_count"] == 1


def test_speaker_labels_zip_when_identical() -> None:
    ref = ["hello", "there"]
    jw = ["hello", "there"]
    js = [1, 2]
    sp = speaker_labels_for_reference_words(ref, jw, js)
    assert sp == [1, 2]


def test_insertion_attributed_to_last_ref_speaker() -> None:
    ref_words = ["a"]
    hyp_words = ["a", "b"]
    speaker_per_ref = [1]
    out = per_speaker_breakdown(ref_words, hyp_words, speaker_per_ref)
    assert out[1]["insertions"] == 1
    assert out[1]["substitutions"] == 0
