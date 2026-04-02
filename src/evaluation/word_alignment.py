"""Word-level alignment between reference and hypothesis (same tokenization as WER)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rapidfuzz.distance import Levenshtein

from src.evaluation.text_normalizer import TextNormalizer

Op = Literal["=", "S", "D", "I"]


@dataclass(frozen=True)
class AlignmentStep:
    op: Op
    ref: str | None
    hyp: str | None


def _editops_to_steps(
    ref_words: list[str],
    hyp_words: list[str],
) -> list[AlignmentStep]:
    """Convert rapidfuzz editops to AlignmentStep sequence (matches, S, I, D).

    Uses the same rapidfuzz backend as ``word_error_breakdown`` so that S/I/D
    counts are guaranteed to be identical to those returned by that function.
    """
    editops = Levenshtein.editops(ref_words, hyp_words)
    steps: list[AlignmentStep] = []
    ri, hi = 0, 0
    for op in editops:
        # Fill any matching words before this operation.
        while ri < op.src_pos and hi < op.dest_pos:
            steps.append(AlignmentStep("=", ref_words[ri], hyp_words[hi]))
            ri += 1
            hi += 1
        if op.tag == "replace":
            steps.append(AlignmentStep("S", ref_words[op.src_pos], hyp_words[op.dest_pos]))
            ri += 1
            hi += 1
        elif op.tag == "delete":
            steps.append(AlignmentStep("D", ref_words[op.src_pos], None))
            ri += 1
        elif op.tag == "insert":
            steps.append(AlignmentStep("I", None, hyp_words[op.dest_pos]))
            hi += 1
    # Trailing matches.
    while ri < len(ref_words) and hi < len(hyp_words):
        steps.append(AlignmentStep("=", ref_words[ri], hyp_words[hi]))
        ri += 1
        hi += 1
    return steps


def align_words(reference: str, hypothesis: str) -> list[AlignmentStep]:
    """Levenshtein alignment on normalized word sequences (matches WER / S-I-D breakdown)."""
    ref_words = TextNormalizer.normalize(reference).split()
    hyp_words = TextNormalizer.normalize(hypothesis).split()
    return _editops_to_steps(ref_words, hyp_words)


def align_word_lists_with_indices(
    ref_words: list[str],
    hyp_words: list[str],
) -> list[tuple[Op, int | None, int | None]]:
    """Same alignment as ``align_words`` but on pre-split word lists; returns (op, ref_idx, hyp_idx)."""
    editops = Levenshtein.editops(ref_words, hyp_words)
    raw: list[tuple[Op, int | None, int | None]] = []
    ri, hi = 0, 0
    for op in editops:
        while ri < op.src_pos and hi < op.dest_pos:
            raw.append(("=", ri, hi))
            ri += 1
            hi += 1
        if op.tag == "replace":
            raw.append(("S", op.src_pos, op.dest_pos))
            ri += 1
            hi += 1
        elif op.tag == "delete":
            raw.append(("D", op.src_pos, None))
            ri += 1
        elif op.tag == "insert":
            raw.append(("I", None, op.dest_pos))
            hi += 1
    while ri < len(ref_words) and hi < len(hyp_words):
        raw.append(("=", ri, hi))
        ri += 1
        hi += 1
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
