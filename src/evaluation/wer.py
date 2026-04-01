from rapidfuzz.distance import Levenshtein

from src.evaluation.text_normalizer import TextNormalizer


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = TextNormalizer.normalize(reference).split()
    hyp_words = TextNormalizer.normalize(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = Levenshtein.distance(ref_words, hyp_words)
    return distance / len(ref_words)
