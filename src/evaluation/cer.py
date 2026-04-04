"""Character Error Rate (CER) with substitution / insertion / deletion counts."""

from __future__ import annotations

from typing import Literal

from src.evaluation.text_normalizer import TextNormalizer

Op = Literal["=", "S", "D", "I"]


def character_error_breakdown(reference: str, hypothesis: str) -> dict[str, float | int]:
    """CER on normalized text; operations at Unicode codepoint (character) level."""
    ref = TextNormalizer.normalize(reference)
    hyp = TextNormalizer.normalize(hypothesis)
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if not ref_chars:
        ins = len(hyp_chars)
        return {
            "cer": 0.0 if ins == 0 else 1.0,
            "substitutions": 0,
            "insertions": ins,
            "deletions": 0,
            "reference_character_count": 0,
        }
    # CR: this is really high complexity, can reach OOM, since there are no limitation, there is no threshold
    # CR: Should we use some c-extension python library ? it uses c and optimized for this matrices operations? 
    # CR: I think you already uses rapidfuzz, why here you reimplemented the logic?

    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    substitutions = 0
    insertions = 0
    deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i - 1] == hyp_chars[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i -= 1
            j -= 1
            continue
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
            continue
        break

    denom = max(1, n)
    cer = (substitutions + insertions + deletions) / denom
    return {
        "cer": cer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "reference_character_count": n,
    }

# CR: Same comment- you do manual matrices operations instead of usign exisiting sdk
def align_char_lists_with_indices(
    ref_chars: list[str],
    hyp_chars: list[str],
) -> list[tuple[Op, int | None, int | None]]:
    """Levenshtein alignment on character lists (same cost model as ``character_error_breakdown``)."""
    n, m = len(ref_chars), len(hyp_chars)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    raw: list[tuple[Op, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_chars[i - 1] == hyp_chars[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            raw.append(("=", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            raw.append(("S", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            raw.append(("D", i - 1, None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            raw.append(("I", None, j - 1))
            j -= 1
        else:
            break
    raw.reverse()
    return raw
