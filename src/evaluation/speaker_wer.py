"""Per-speaker WER (and S/I/D) using gold speaker labels from transcribed JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.text_normalizer import TextNormalizer
from src.evaluation.wer import word_error_breakdown
from src.evaluation.word_alignment import align_word_lists_with_indices

from src.evaluation.transcribed_json import load_flat_words_with_speakers


def _clamp_speaker(sp: int) -> int:
    if sp in (1, 2):
        return sp
    return 1


def speaker_labels_for_reference_words(
    ref_words: list[str],
    json_words: list[str],
    json_speakers: list[int],
) -> list[int]:
    """Assign each reference word a speaker id (1 or 2) using alignment to JSON words."""
    if len(json_words) != len(json_speakers):
        return [1] * len(ref_words)
    if ref_words == json_words:
        return [_clamp_speaker(s) for s in json_speakers]

    steps = align_word_lists_with_indices(ref_words, json_words)
    speakers = [1] * len(ref_words)
    last_sp = 1
    for op, ri, hj in steps:
        if ri is not None and hj is not None:
            speakers[ri] = _clamp_speaker(json_speakers[hj])
            last_sp = speakers[ri]
        elif op == "D" and ri is not None:
            speakers[ri] = last_sp
    return speakers


def per_speaker_breakdown(
    ref_words: list[str],
    hyp_words: list[str],
    speaker_per_ref: list[int],
) -> dict[int, dict[str, Any]]:
    """Partition S/I/D from the gold-vs-hyp alignment into speakers 1 and 2."""
    steps = align_word_lists_with_indices(ref_words, hyp_words)
    S = {1: 0, 2: 0}
    I = {1: 0, 2: 0}
    D = {1: 0, 2: 0}
    ref_count = {1: 0, 2: 0}
    for sp in speaker_per_ref:
        if sp in ref_count:
            ref_count[sp] += 1

    last_sp = 1
    for op, ri, hj in steps:
        if ri is not None:
            last_sp = speaker_per_ref[ri]
        if op == "=":
            continue
        if op == "S":
            S[last_sp] += 1
        elif op == "D":
            D[last_sp] += 1
        elif op == "I":
            I[last_sp] += 1

    out: dict[int, dict[str, Any]] = {}
    for sp in (1, 2):
        rc = ref_count[sp]
        errs = S[sp] + I[sp] + D[sp]
        wer_s = (errs / rc) if rc > 0 else None
        out[sp] = {
            "wer": wer_s,
            "substitutions": S[sp],
            "insertions": I[sp],
            "deletions": D[sp],
            "reference_word_count": rc,
        }
    return out


def compute_speaker_wer_for_sample(
    *,
    gold_text: str,
    hypothesis_text: str,
    transcribed_json_path: Path,
) -> dict[str, Any] | None:
    """If ``transcribed_json_path`` exists, return overall + per-speaker metrics; else None."""
    loaded = load_flat_words_with_speakers(transcribed_json_path)
    if loaded is None:
        return None
    json_words, json_speakers = loaded
    ref_words = TextNormalizer.normalize(gold_text).split()
    if not ref_words:
        return None

    speaker_per_ref = speaker_labels_for_reference_words(ref_words, json_words, json_speakers)
    hyp_words = TextNormalizer.normalize(hypothesis_text).split()

    overall = word_error_breakdown(gold_text, hypothesis_text)
    by_sp = per_speaker_breakdown(ref_words, hyp_words, speaker_per_ref)
    return {
        "overall": overall,
        "speaker_1": by_sp[1],
        "speaker_2": by_sp[2],
        "speaker_labels_source": str(transcribed_json_path),
    }
