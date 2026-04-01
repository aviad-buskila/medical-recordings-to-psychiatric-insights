import logging
from pathlib import Path

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.judge import OllamaJudge
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader

logger = logging.getLogger(__name__)


def run_llm_judge_eval(
    run_id: str,
    ref_run_id: str | None = None,
    limit: int | None = None,
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

    candidate_outputs = analytics.get_stt_outputs_for_run(run_id)
    baseline_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}

    sample_ids = list(candidate_outputs.keys())
    if ref_run_id:
        sample_ids = [sid for sid in sample_ids if sid in baseline_outputs]

    evaluated = 0
    wins_candidate = 0
    wins_baseline = 0
    ties = 0
    score_deltas: list[float] = []
    parse_failures = 0
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
            rationale_short = (rationale[:140] + "...") if len(rationale) > 140 else rationale

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
                "Sample %s | winner=%s | candidate=%.2f baseline=%.2f delta=%+.2f | rationale=%s",
                sample_id,
                winner,
                candidate_score,
                baseline_score,
                delta_value,
                rationale_short or "n/a",
            )
            analytics.insert_eval_metric(
                sample_id=sample_id,
                metric_name="llm_judge_compare",
                metric_value=delta_value,
                details={"run_id": run_id, "ref_run_id": ref_run_id, "judge_result": result},
            )
        else:
            result = judge.evaluate_transcript(reference, candidate_text)
            score = float(result.get("overall_score", 0.0) or 0.0)
            logger.info(
                "Sample %s | overall=%.2f | del=%s ins=%s sub=%s risk=%s",
                sample_id,
                score,
                result.get("deletion_error_severity", "n/a"),
                result.get("insertion_error_severity", "n/a"),
                result.get("substitution_error_severity", "n/a"),
                result.get("medical_safety_risk", "n/a"),
            )
            analytics.insert_eval_metric(
                sample_id=sample_id,
                metric_name="llm_judge_score",
                metric_value=score,
                details={"run_id": run_id, "judge_result": result},
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


def _resolve_gold_reference(sample_id: str, gold_transcripts: dict[str, str], transcript_path: Path | None) -> str | None:
    candidates = [sample_id, sample_id.upper(), sample_id.replace("_", "-"), sample_id.replace("-", "_")]
    for candidate in candidates:
        if candidate in gold_transcripts:
            return gold_transcripts[candidate]
    if transcript_path and transcript_path.exists():
        return Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    return None
