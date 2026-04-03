"""Tests for cp wer behavior."""

import json
from pathlib import Path

from src.evaluation.cp_wer import cp_wer_breakdown_from_json


def test_cp_wer_prefers_block_order_when_hyp_matches(tmp_path: Path) -> None:
    j = tmp_path / "S.json"
    j.write_text(
        json.dumps(
            [
                {"speaker": 1, "dialogue": ["hello"]},
                {"speaker": 2, "dialogue": ["world"]},
            ]
        ),
        encoding="utf-8",
    )
    # Chronological "hello world"; blocks 2|1 => "world hello" — hypothesis matches that order
    out = cp_wer_breakdown_from_json("world hello", j)
    assert out is not None
    assert out["cp_wer"] == 0.0
    assert "blocks_order" in out["best_candidate_label"]


def test_cp_wer_single_speaker_matches_wer(tmp_path: Path) -> None:
    j = tmp_path / "S.json"
    j.write_text(
        json.dumps([{"speaker": 1, "dialogue": ["only", "one"]}]),
        encoding="utf-8",
    )
    out = cp_wer_breakdown_from_json("only one", j)
    assert out is not None
    assert out["cp_wer"] == 0.0
