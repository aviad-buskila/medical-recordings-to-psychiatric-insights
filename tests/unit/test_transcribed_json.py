"""Tests for transcribed json behavior."""

import json

from src.evaluation.transcribed_json import load_flat_words_with_speakers, transcribed_json_path


def test_transcribed_json_path_layout() -> None:
    p = transcribed_json_path("/data/transcripts", "S1-T01")
    assert p.name == "S1-T01.json"
    assert "transcribed" in p.parts


def test_load_flat_words_with_speakers_from_json(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(
        json.dumps(
            [
                {"speaker": 1, "dialogue": ["Hello world"]},
                {"speaker": 2, "dialogue": ["Bye"]},
            ]
        ),
        encoding="utf-8",
    )
    out = load_flat_words_with_speakers(path)
    assert out is not None
    words, speakers = out
    assert words == ["hello", "world", "bye"]
    assert speakers == [1, 1, 2]


def test_load_flat_words_missing_file(tmp_path) -> None:
    assert load_flat_words_with_speakers(tmp_path / "missing.json") is None


def test_load_flat_words_invalid_not_list(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert load_flat_words_with_speakers(path) is None
