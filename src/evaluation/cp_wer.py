"""Concatenated minimum-Permutation Word Error Rate (cpWER).

For multi-speaker references (from ``transcripts/transcribed/*.json``), cpWER is the
minimum word-level error rate over several **reference** strings built from the same
JSON:

1. **Chronological**: words in dialogue order (typical match for ``dataset.pickle``).
2. **Block permutations**: for each ordering π of speaker ids, concatenate each
   speaker's words (in dialogue order) as ``block_π(1) || block_π(2) || …``.

The hypothesis is the full ASR string (no speaker labels). The minimum is taken over
normalized WER edit counts.

If only one distinct speaker appears, candidates collapse and cpWER matches WER on that
reference. If more than ``MAX_SPEAKERS_CP_WER`` distinct speakers appear, evaluation
returns ``None`` (factorial cost cap).
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any

from src.evaluation.text_normalizer import TextNormalizer
from src.evaluation.transcribed_json import load_flat_words_with_speakers
from src.evaluation.wer import word_error_breakdown

logger = logging.getLogger(__name__)

MAX_SPEAKERS_CP_WER = 8


def _aggregate_words_by_speaker(words: list[str], speakers: list[int]) -> dict[int, list[str]]:
    by_sp: dict[int, list[str]] = {}
    for w, s in zip(words, speakers, strict=True):
        by_sp.setdefault(s, []).append(w)
    return by_sp


def cp_wer_breakdown_from_json(
    hypothesis_text: str,
    transcribed_json_path: Path,
) -> dict[str, Any] | None:
    """Return cpWER and the winning WER breakdown; ``None`` if JSON missing or unusable."""
    loaded = load_flat_words_with_speakers(transcribed_json_path)
    if not loaded:
        return None
    words, speakers = loaded
    if not words:
        return None

    speaker_ids = sorted(set(speakers))
    if len(speaker_ids) > MAX_SPEAKERS_CP_WER:
        logger.warning(
            "cpWER skipped: %s distinct speakers exceeds max %s",
            len(speaker_ids),
            MAX_SPEAKERS_CP_WER,
        )
        return None

    by_sp = _aggregate_words_by_speaker(words, speakers)

    labeled_candidates: list[tuple[str, str]] = []
    labeled_candidates.append(("chronological", " ".join(words)))
    # CR: you go bruteforce here - can be veru slow o(n!)? cannot find a better algo?
    for perm in itertools.permutations(speaker_ids):
        label = "blocks_order:" + ",".join(str(s) for s in perm)
        parts = [" ".join(by_sp[s]) for s in perm]
        labeled_candidates.append((label, " ".join(parts)))

    seen_norm: set[str] = set()
    unique: list[tuple[str, str]] = []
    for label, ref in labeled_candidates:
        norm = TextNormalizer.normalize(ref)
        if norm not in seen_norm:
            seen_norm.add(norm)
            unique.append((label, ref))

    best_bd: dict[str, Any] | None = None
    best_dist: int | None = None
    best_label: str | None = None
    best_ref_preview: str | None = None

    for label, ref in unique:
        bd = word_error_breakdown(ref, hypothesis_text)
        dist = int(bd["substitutions"] + bd["insertions"] + bd["deletions"])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_bd = bd
            best_label = label
            best_ref_preview = TextNormalizer.normalize(ref)[:400]

    assert best_bd is not None and best_label is not None
    return {
        "cp_wer": float(best_bd["wer"]),
        "cp_wer_breakdown": best_bd,
        "best_reference_preview": best_ref_preview,
        "best_candidate_label": best_label,
        "candidates_evaluated": len(unique),
        "block_permutations_generated": math.factorial(len(speaker_ids)),
        "distinct_speakers": speaker_ids,
    }
