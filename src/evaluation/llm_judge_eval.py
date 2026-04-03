"""Evaluation utilities for llm judge eval."""

import logging
from pathlib import Path
from typing import Any

import psycopg

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.judge import OllamaJudge
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader
from src.core.eval_run_report import EvalRunReporter

logger = logging.getLogger(__name__)


def run_llm_judge_eval(
    run_id: str,
    ref_run_id: str | None = None,
    sample_id: str | None = None,
    limit: int | None = None,
    reporter: EvalRunReporter | None = None,
) -> None:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    samples = loader.load_samples()
    samples_by_id = {s.sample_id: s for s in samples}

    pickle_loader = DatasetPickleLoader(settings.dataset_pickle_path)
    gold_transcripts = pickle_loader.load_transcripts()

    analytics = AnalyticsRepository()
    judge = OllamaJudge()
    candidate_run_info = analytics.get_stt_run_info(run_id)
    baseline_run_info = analytics.get_stt_run_info(ref_run_id) if ref_run_id else None
    if reporter:
        reporter.set_run_metadata(
            run_id=run_id,
            ref_run_id=ref_run_id,
            sample_id=sample_id,
            candidate_run_info=candidate_run_info,
            ref_run_info=baseline_run_info,
        )

    candidate_outputs = analytics.get_stt_outputs_for_run(run_id)
    baseline_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}

    sample_ids = list(candidate_outputs.keys())
    if ref_run_id:
        # In compare mode, require both candidate and baseline transcripts.
        sample_ids = [sid for sid in sample_ids if sid in baseline_outputs]
    if sample_id:
        sample_ids = [sid for sid in sample_ids if sid == sample_id]

    evaluated = 0
    wins_candidate = 0
    wins_baseline = 0
    ties = 0
    score_deltas: list[float] = []
    parse_failures = 0
    llm_failures = 0
    candidate_label = (
        f"{candidate_run_info.get('provider')}:{candidate_run_info.get('model_name')}"
        if candidate_run_info
        else run_id
    )
    baseline_label = (
        f"{baseline_run_info.get('provider')}:{baseline_run_info.get('model_name')}"
        if baseline_run_info
        else (ref_run_id or "")
    )
    if ref_run_id:
        logger.info(
            "LLM judge compare setup | candidate_run=%s (%s) vs baseline_run=%s (%s) | score_scale=0-10 (higher is better)",
            run_id,
            candidate_label,
            ref_run_id,
            baseline_label,
        )
    else:
        logger.info(
            "LLM judge setup | run=%s (%s) | score_scale=0-10 (higher is better)",
            run_id,
            candidate_label,
        )
    for sample_id in sample_ids:
        if limit is not None and evaluated >= limit:
            break
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        reference = _resolve_gold_reference(sample_id, gold_transcripts, sample.transcript_path)
        if not reference:
            continue

        candidate_text = candidate_outputs.get(sample_id, "")
        if not candidate_text:
            continue

        # Avoid duplicate inserts when resuming after a failure: if the exact metric row
        # already exists for (sample_id, run_id, ref_run_id), reuse it.
        metric_name = "llm_judge_compare" if ref_run_id else "llm_judge_score"
        existing = _load_existing_llm_judge_metric(
            postgres_dsn=analytics.settings.postgres_dsn,
            sample_id=sample_id,
            metric_name=metric_name,
            run_id=run_id,
            ref_run_id=ref_run_id or "",
        )

        if ref_run_id:
            baseline_text = baseline_outputs.get(sample_id, "")
            if not baseline_text:
                continue
            if existing is not None:
                # Resume path: reuse prior DB value, but still include in report output.
                result = existing.get("judge_result", {}) or {}
                winner = str(result.get("winner", "unknown")).lower()
                delta_value = float(existing.get("metric_value", 0.0) or 0.0)
                candidate_score = float(result.get("candidate_overall_score", 0.0) or 0.0)
                baseline_score = float(result.get("baseline_overall_score", 0.0) or 0.0)
                rationale = str(result.get("rationale", "")).strip()
            else:
                try:
                    result = judge.compare_transcripts(reference, candidate_text, baseline_text)
                except Exception as e:
                    # Soft-fail so long runs continue and still produce a complete artifact.
                    llm_failures += 1
                    result = {
                        "winner": "unknown",
                        "candidate_overall_score": 0.0,
                        "baseline_overall_score": 0.0,
                        "score_delta": 0.0,
                        "rationale": "LLM judge failed (timeout/error).",
                        "judge_error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        },
                    }

                winner = str(result.get("winner", "unknown")).lower()
                delta_value = float(result.get("score_delta", 0.0) or 0.0)
                candidate_score = float(result.get("candidate_overall_score", 0.0) or 0.0)
                baseline_score = float(result.get("baseline_overall_score", 0.0) or 0.0)
                rationale = str(result.get("rationale", "")).strip()

            if winner == "candidate":
                wins_candidate += 1
            elif winner == "baseline":
                wins_baseline += 1
            elif winner == "tie":
                ties += 1
            else:
                parse_failures += 1
            score_deltas.append(delta_value)

            logger.info(
                "Sample %s | winner=%s | candidate[%s]=%.2f/10 baseline[%s]=%.2f/10 delta(candidate-baseline)=%+.2f | rationale=%s",
                sample_id,
                winner,
                candidate_label,
                candidate_score,
                baseline_label,
                baseline_score,
                delta_value,
                rationale or "n/a",
            )
            if existing is None:
                details = {
                    "run_id": run_id,
                    "ref_run_id": ref_run_id,
                    "judge_result": result,
                }
                # Persist exactly one logical row per sample/run pair.
                analytics.insert_eval_metric(
                    sample_id=sample_id,
                    metric_name="llm_judge_compare",
                    metric_value=delta_value,
                    details=details,
                )
                if reporter:
                    reporter.add_metric(
                        sample_id=sample_id,
                        metric_name="llm_judge_compare",
                        metric_value=delta_value,
                        details=details,
                    )
            else:
                # Still add to report file to keep run-llm-judge output consistent.
                if reporter:
                    details = {"run_id": run_id, "ref_run_id": ref_run_id, "judge_result": result}
                    reporter.add_metric(
                        sample_id=sample_id,
                        metric_name="llm_judge_compare",
                        metric_value=delta_value,
                        details=details,
                    )
        else:
            if existing is not None:
                # Resume path for single-run scoring mode.
                result = existing.get("judge_result", {}) or {}
                score = float(existing.get("metric_value", 0.0) or 0.0)
            else:
                try:
                    result = judge.evaluate_transcript(reference, candidate_text)
                except Exception as e:
                    llm_failures += 1
                    result = {
                        "overall_score": 0.0,
                        "rationale": "LLM judge failed (timeout/error).",
                        "judge_error": {
                            "type": type(e).__name__,
                            "message": str(e),
                        },
                    }
                score = float(result.get("overall_score", 0.0) or 0.0)

            logger.info(
                "Sample %s | overall[%s]=%.2f/10 | del=%s/10 ins=%s/10 sub=%s/10 risk=%s/10",
                sample_id,
                candidate_label,
                score,
                result.get("deletion_error_severity", "n/a"),
                result.get("insertion_error_severity", "n/a"),
                result.get("substitution_error_severity", "n/a"),
                result.get("medical_safety_risk", "n/a"),
            )
            if existing is None:
                details = {"run_id": run_id, "judge_result": result}
                analytics.insert_eval_metric(
                    sample_id=sample_id,
                    metric_name="llm_judge_score",
                    metric_value=score,
                    details=details,
                )
                if reporter:
                    reporter.add_metric(
                        sample_id=sample_id,
                        metric_name="llm_judge_score",
                        metric_value=score,
                        details=details,
                    )
            else:
                if reporter:
                    details = {"run_id": run_id, "judge_result": result}
                    reporter.add_metric(
                        sample_id=sample_id,
                        metric_name="llm_judge_score",
                        metric_value=score,
                        details=details,
                    )
        evaluated += 1

    if ref_run_id and evaluated > 0:
        avg_delta = sum(score_deltas) / len(score_deltas) if score_deltas else 0.0
        logger.info(
            "LLM judge compare summary | evaluated=%s candidate_wins=%s baseline_wins=%s ties=%s parse_failures=%s avg_delta=%+.3f",
            evaluated,
            wins_candidate,
            wins_baseline,
            ties,
            parse_failures,
            avg_delta,
        )
    logger.info("LLM judge failures summary | llm_failures=%s evaluated=%s", llm_failures, evaluated)

    logger.info(
        "LLM judge evaluation completed. run_id=%s ref_run_id=%s evaluated=%s",
        run_id,
        ref_run_id,
        evaluated,
    )

    if reporter:
        reporter.set_result_summary(
            evaluated=evaluated,
            wins_candidate=wins_candidate if ref_run_id else None,
            wins_baseline=wins_baseline if ref_run_id else None,
            ties=ties if ref_run_id else None,
            parse_failures=parse_failures,
            wins_candidate_only=wins_candidate if not ref_run_id else None,
        )


def _resolve_gold_reference(sample_id: str, gold_transcripts: dict[str, str], transcript_path: Path | None) -> str | None:
    candidates = [sample_id, sample_id.upper(), sample_id.replace("_", "-"), sample_id.replace("-", "_")]
    for candidate in candidates:
        if candidate in gold_transcripts:
            return gold_transcripts[candidate]
    if transcript_path and transcript_path.exists():
        return Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    return None


def _load_existing_llm_judge_metric(
    *,
    postgres_dsn: str,
    sample_id: str,
    metric_name: str,
    run_id: str,
    ref_run_id: str,
) -> dict[str, Any] | None:
    """
    Returns existing judge metric info to support resuming without duplicate inserts.

    We match by:
      - sample_id
      - metric_name
      - details.run_id
      - details.ref_run_id (empty string if absent)
    """
    with psycopg.connect(postgres_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT metric_value, details->'judge_result'
                FROM clinical_ai.evaluation_metrics
                WHERE sample_id = %(sample_id)s
                  AND metric_name = %(metric_name)s
                  AND details->>'run_id' = %(run_id)s
                  AND COALESCE(details->>'ref_run_id', '') = %(ref_run_id)s
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                {
                    "sample_id": sample_id,
                    "metric_name": metric_name,
                    "run_id": run_id,
                    "ref_run_id": ref_run_id,
                },
            )
            row = cur.fetchone()
            if not row:
                return None
            metric_value = float(row[0] or 0.0)
            judge_result = row[1] if isinstance(row[1], dict) else {}
            return {"metric_value": metric_value, "judge_result": judge_result}
