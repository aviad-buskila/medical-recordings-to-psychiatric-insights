from rapidfuzz.distance import Levenshtein


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words)
