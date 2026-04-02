"""Word-level alignment between reference and hypothesis (same tokenization as WER)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.evaluation.text_normalizer import TextNormalizer

Op = Literal["=", "S", "D", "I"]


@dataclass(frozen=True)
class AlignmentStep:
    op: Op
    ref: str | None
    hyp: str | None


def align_words(reference: str, hypothesis: str) -> list[AlignmentStep]:
    """Levenshtein alignment on normalized word sequences (matches WER / S-I-D breakdown)."""
    ref_words = TextNormalizer.normalize(reference).split()
    hyp_words = TextNormalizer.normalize(hypothesis).split()
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    # Backtrace mirrors ``word_error_breakdown`` in ``wer.py`` so S/I/D counts stay consistent.
    steps: list[AlignmentStep] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            steps.append(AlignmentStep("=", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            steps.append(AlignmentStep("S", ref_words[i - 1], hyp_words[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            steps.append(AlignmentStep("D", ref_words[i - 1], None))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            steps.append(AlignmentStep("I", None, hyp_words[j - 1]))
            j -= 1
        else:
            break

    steps.reverse()
    return steps


def align_word_lists_with_indices(
    ref_words: list[str],
    hyp_words: list[str],
) -> list[tuple[Op, int | None, int | None]]:
    """Same alignment as ``align_words`` but on pre-split word lists; returns (op, ref_idx, hyp_idx)."""
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    raw: list[tuple[Op, int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
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


def format_alignment_table(steps: list[AlignmentStep], *, include_legend: bool = True) -> str:
    """Side-by-side REF / HYP / OP with column padding (plain text)."""
    if not steps:
        return "(empty alignment)\n"

    gap = "·"
    ref_cells = [s.ref if s.ref is not None else gap for s in steps]
    hyp_cells = [s.hyp if s.hyp is not None else gap for s in steps]
    op_cells = [s.op for s in steps]

    widths: list[int] = []
    for a, b, c in zip(ref_cells, hyp_cells, op_cells, strict=True):
        widths.append(max(len(a), len(b), len(c)))

    def row(label: str, cells: list[str]) -> str:
        parts = [cells[k].ljust(widths[k]) for k in range(len(cells))]
        return label + "  ".join(parts)

    lines = [
        row("GOLD ", ref_cells),
        row("HYP  ", hyp_cells),
        row("OP   ", op_cells),
    ]
    out = "\n".join(lines) + "\n"
    if include_legend:
        out += "(= match, S substitution, D deletion from gold, I insertion vs gold)\n"
    return out


def count_ops(steps: list[AlignmentStep]) -> dict[str, int]:
    return {
        "substitutions": sum(1 for s in steps if s.op == "S"),
        "insertions": sum(1 for s in steps if s.op == "I"),
        "deletions": sum(1 for s in steps if s.op == "D"),
        "matches": sum(1 for s in steps if s.op == "="),
    }
