"""Print word-level alignment: gold (dataset.pickle) vs STT run hypothesis(es)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.text_normalizer import TextNormalizer
from src.evaluation.wer import word_error_breakdown
from src.evaluation.word_alignment import align_words, count_ops, format_alignment_table
from src.ingestion.pickle_loader import DatasetPickleLoader

logger = logging.getLogger(__name__)


def run_alignment_report(
    run_id: str,
    ref_run_id: str | None = None,
    limit: int | None = None,
    sample_ids: tuple[str, ...] | None = None,
    chunk_columns: int = 48,
    output_path: str | None = None,
) -> str:
    """Build a text report; optionally write to ``output_path``."""
    settings = get_settings()
    pickle_loader = DatasetPickleLoader(settings.dataset_pickle_path)
    gold_transcripts = pickle_loader.load_transcripts()
    if not gold_transcripts:
        msg = f"No gold transcripts found in {settings.dataset_pickle_path}"
        logger.warning(msg)
        return msg + "\n"

    analytics = AnalyticsRepository()
    candidate_outputs = analytics.get_stt_outputs_for_run(run_id)
    baseline_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}

    cand_info = analytics.get_stt_run_info(run_id)
    base_info = analytics.get_stt_run_info(ref_run_id) if ref_run_id else None

    ids = list(candidate_outputs.keys())
    if ref_run_id:
        ids = [sid for sid in ids if sid in baseline_outputs]
    if sample_ids:
        wanted = set(sample_ids)
        ids = [sid for sid in ids if sid in wanted]
    ids = [sid for sid in ids if sid in gold_transcripts]

    if limit is not None:
        ids = ids[:limit]

    lines: list[str] = []
    header = _header_block(run_id, ref_run_id, cand_info, base_info)
    lines.append(header)

    for sample_id in ids:
        gold_raw = gold_transcripts.get(sample_id, "")
        hyp_a = candidate_outputs.get(sample_id, "")
        block = _sample_block(
            sample_id=sample_id,
            gold_raw=gold_raw,
            hyp_primary=hyp_a,
            hyp_secondary=baseline_outputs.get(sample_id) if ref_run_id else None,
            primary_label=f"run {run_id[:8]}…",
            secondary_label=f"run {ref_run_id[:8]}…" if ref_run_id else None,
            chunk_columns=chunk_columns,
        )
        lines.append(block)

    if not ids:
        lines.append("No samples to align (check run_id, gold keys, and optional --sample-id).\n")

    text = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(text, encoding="utf-8")
        logger.info("Wrote alignment report to %s", output_path)
    return text


def _header_block(
    run_id: str,
    ref_run_id: str | None,
    cand: dict | None,
    base: dict | None,
) -> str:
    parts = [
        "=" * 80,
        "WORD ALIGNMENT (gold = dataset.pickle transcripts, same normalization as WER)",
        f"Candidate run_id: {run_id}",
    ]
    if cand:
        parts.append(f"  model: {cand.get('model_name')}  scope: {cand.get('run_scope')}")
    if ref_run_id:
        parts.append(f"Baseline run_id:  {ref_run_id}")
        if base:
            parts.append(f"  model: {base.get('model_name')}  scope: {base.get('run_scope')}")
    parts.append("=" * 80)
    return "\n".join(parts) + "\n"


def _sample_block(
    sample_id: str,
    gold_raw: str,
    hyp_primary: str,
    hyp_secondary: str | None,
    primary_label: str,
    secondary_label: str | None,
    chunk_columns: int,
) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"sample_id: {sample_id}")
    lines.append("-" * 80)

    # Primary vs gold
    br = word_error_breakdown(gold_raw, hyp_primary)
    lines.append(
        f"WER vs gold ({primary_label}): {br['wer']:.4f}  "
        f"S={br['substitutions']} I={br['insertions']} D={br['deletions']}  "
        f"(ref words={br['reference_word_count']})"
    )
    steps = align_words(gold_raw, hyp_primary)
    ops = count_ops(steps)
    lines.append(f"counts: matches={ops['matches']}")
    lines.extend(_chunked_alignment_lines(steps, chunk_columns))

    if hyp_secondary is not None and secondary_label is not None:
        lines.append("")
        br2 = word_error_breakdown(gold_raw, hyp_secondary)
        lines.append(
            f"WER vs gold ({secondary_label}): {br2['wer']:.4f}  "
            f"S={br2['substitutions']} I={br2['insertions']} D={br2['deletions']}  "
            f"(ref words={br2['reference_word_count']})"
        )
        steps2 = align_words(gold_raw, hyp_secondary)
        ops2 = count_ops(steps2)
        lines.append(f"counts: matches={ops2['matches']}")
        lines.extend(_chunked_alignment_lines(steps2, chunk_columns))

    # Short raw preview (optional context)
    g_norm = TextNormalizer.normalize(gold_raw)
    if len(g_norm) > 240:
        lines.append("")
        lines.append(f"gold (normalized, truncated): {g_norm[:240]}…")
    return "\n".join(lines) + "\n"


def _chunked_alignment_lines(steps: list, chunk_columns: int) -> list[str]:
    if chunk_columns < 8:
        chunk_columns = 48
    lines: list[str] = []
    if not steps:
        lines.append("(no words after normalization)")
        return lines

    for start in range(0, len(steps), chunk_columns):
        chunk = steps[start : start + chunk_columns]
        is_last = start + chunk_columns >= len(steps)
        table = format_alignment_table(chunk, include_legend=is_last)
        lines.extend(table.rstrip().splitlines())
        if not is_last:
            lines.append(f"… columns {start + 1}–{start + len(chunk)} of {len(steps)} …")
    return lines
