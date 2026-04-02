import logging
from pathlib import Path

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
        sample_ids = [sid for sid in sample_ids if sid in baseline_outputs]
    if sample_id:
        sample_ids = [sid for sid in sample_ids if sid == sample_id]

    evaluated = 0
    wins_candidate = 0
    wins_baseline = 0
    ties = 0
    score_deltas: list[float] = []
    parse_failures = 0
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

        if ref_run_id:
            baseline_text = baseline_outputs.get(sample_id, "")
            if not baseline_text:
                continue
            result = judge.compare_transcripts(reference, candidate_text, baseline_text)
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
            details = {
                "run_id": run_id,
                "ref_run_id": ref_run_id,
                "judge_result": result,
            }
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
            result = judge.evaluate_transcript(reference, candidate_text)
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
