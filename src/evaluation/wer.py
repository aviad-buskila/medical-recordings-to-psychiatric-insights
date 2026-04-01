from rapidfuzz.distance import Levenshtein

from src.evaluation.text_normalizer import TextNormalizer


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = TextNormalizer.normalize(reference).split()
    hyp_words = TextNormalizer.normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words)


def word_error_breakdown(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Return WER and operation-level counts (S/I/D)."""
    ref_words = TextNormalizer.normalize(reference).split()
    hyp_words = TextNormalizer.normalize(hypothesis).split()

    n = len(ref_words)
    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution/match
            )

    substitutions = 0
    insertions = 0
    deletions = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
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

    denominator = max(1, n)
    wer = (substitutions + insertions + deletions) / denominator
    return {
        "wer": wer,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "reference_word_count": n,
        "substitution_rate": substitutions / denominator,
        "insertion_rate": insertions / denominator,
        "deletion_rate": deletions / denominator,
    }
