"""Unit tests for gold reference resolution (no DB)."""

from pathlib import Path

from src.evaluation.stt_eval import _resolve_gold_reference


def test_resolve_from_pickle_map_exact_id() -> None:
    gold = {"ABC-01": "ref text"}
    assert _resolve_gold_reference("ABC-01", gold, None) == "ref text"


def test_resolve_from_pickle_map_variant_underscore() -> None:
    gold = {"A_B": "x"}
    assert _resolve_gold_reference("A-B", gold, None) == "x"


def test_resolve_missing_returns_none(tmp_path: Path) -> None:
    assert _resolve_gold_reference("nope", {}, None) is None


def test_resolve_from_transcript_file(tmp_path: Path) -> None:
    p = tmp_path / "t.txt"
    p.write_text("from file", encoding="utf-8")
    assert _resolve_gold_reference("any", {}, p) == "from file"
