"""Load speaker-segmented transcripts from ``data/raw/transcripts/transcribed/<id>.json``."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.text_normalizer import TextNormalizer


def transcribed_json_path(transcripts_dir: str, sample_id: str) -> Path:
    return Path(transcripts_dir) / "transcribed" / f"{sample_id}.json"


def load_flat_words_with_speakers(json_path: Path) -> tuple[list[str], list[int]] | None:
    """Return normalized words and parallel speaker ids (1, 2, …) from transcribed JSON."""
    if not json_path.exists():
        return None
    raw = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    if not isinstance(raw, list):
        return None
    words: list[str] = []
    speakers: list[int] = []
    for block in raw:
        if not isinstance(block, dict):
            continue
        sp_raw = block.get("speaker")
        dialogue = block.get("dialogue")
        if sp_raw is None or not isinstance(dialogue, list):
            continue
        try:
            sp = int(sp_raw)
        except (TypeError, ValueError):
            continue
        for line in dialogue:
            if not isinstance(line, str):
                continue
            for w in TextNormalizer.normalize(line).split():
                if w:
                    words.append(w)
                    speakers.append(sp)
    if not words:
        return None
    return words, speakers
