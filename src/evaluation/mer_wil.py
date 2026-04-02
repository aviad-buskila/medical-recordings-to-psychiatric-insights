"""Match Error Rate (MER) and Word Information Lost (WIL) at word level.

Definitions (aligned with common ASR reporting, e.g. jiwer-style):

- R = |reference words|, H = |hypothesis words| (after same normalization as WER).
- E = S + I + D (total edit operations from the WER alignment).
- MER = E / (R + H)   (if R + H == 0, MER = 0).
- WIL = 2 * E / (R + H) = 2 * MER.

Per-speaker: same formulas with E_s, R_s, H_s from the gold–hypothesis alignment.
"""

from __future__ import annotations

from src.evaluation.text_normalizer import TextNormalizer
from src.evaluation.wer import word_error_breakdown


def word_mer_wil_breakdown(reference: str, hypothesis: str) -> dict[str, float | int]:
    bd = word_error_breakdown(reference, hypothesis)
    r = int(bd["reference_word_count"])
    h = len(TextNormalizer.normalize(hypothesis).split())
    s = int(bd["substitutions"])
    ins = int(bd["insertions"])
    d = int(bd["deletions"])
    e = s + ins + d
    denom = r + h
    if denom == 0:
        return {
            "mer": 0.0,
            "wil": 0.0,
            "substitutions": s,
            "insertions": ins,
            "deletions": d,
            "edit_total": e,
            "reference_word_count": r,
            "hypothesis_word_count": h,
        }
    mer = e / denom
    wil = 2.0 * e / denom
    return {
        "mer": mer,
        "wil": wil,
        "substitutions": s,
        "insertions": ins,
        "deletions": d,
        "edit_total": e,
        "reference_word_count": r,
        "hypothesis_word_count": h,
    }


def mer_wil_from_counts(
    *,
    substitutions: int,
    insertions: int,
    deletions: int,
    reference_word_count: int,
    hypothesis_word_count: int,
) -> dict[str, float | int | None]:
    e = substitutions + insertions + deletions
    denom = reference_word_count + hypothesis_word_count
    if denom == 0:
        return {"mer": None, "wil": None, "edit_total": e}
    return {
        "mer": e / denom,
        "wil": 2.0 * e / denom,
        "edit_total": e,
    }
