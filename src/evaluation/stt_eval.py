import logging
from pathlib import Path

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.wer import word_error_rate
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader

logger = logging.getLogger(__name__)


def evaluate_stt_against_gold(
    limit: int | None = None,
    run_id: str | None = None,
    ref_run_id: str | None = None,
) -> None:
    """Compare latest STT outputs against gold transcripts using WER.

    Gold references are sourced primarily from dataset.pickle.
    """
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    pickle_loader = DatasetPickleLoader(settings.dataset_pickle_path)
    gold_transcripts = pickle_loader.load_transcripts()
    analytics = AnalyticsRepository()
    samples = loader.load_samples()
    samples_by_id = {s.sample_id: s for s in samples}
    run_outputs = analytics.get_stt_outputs_for_run(run_id) if run_id else {}
    ref_run_outputs = analytics.get_stt_outputs_for_run(ref_run_id) if ref_run_id else {}

    evaluated = 0
    if run_id:
        candidate_sample_ids = list(run_outputs.keys())
        if ref_run_id:
            candidate_sample_ids = [sid for sid in candidate_sample_ids if sid in ref_run_outputs]
    else:
        candidate_sample_ids = [s.sample_id for s in samples]

    for sample_id in candidate_sample_ids:
        if limit is not None and evaluated >= limit:
            break
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        reference = _resolve_gold_reference(sample.sample_id, gold_transcripts, sample.transcript_path)
        if not reference:
            continue
        hypothesis = run_outputs.get(sample.sample_id) if run_id else analytics.get_latest_stt_output(sample.sample_id)
        hypothesis = hypothesis or ""
        if not hypothesis:
            if not run_id:
                logger.warning("No STT output found in DB for sample_id=%s", sample.sample_id)
            continue
        wer = word_error_rate(reference=reference, hypothesis=hypothesis)
        details = {"reference_source": "dataset.pickle", "run_id": run_id}
        if ref_run_id:
            ref_hypothesis = ref_run_outputs.get(sample.sample_id, "")
            if not ref_hypothesis:
                continue
            ref_wer = word_error_rate(reference=reference, hypothesis=ref_hypothesis)
            delta = wer - ref_wer
            details.update({"ref_run_id": ref_run_id, "ref_wer": ref_wer, "delta_vs_ref": delta})
            logger.info(
                "Sample %s WER run=%s: %.4f | ref=%s: %.4f | delta=%.4f",
                sample.sample_id,
                run_id,
                wer,
                ref_run_id,
                ref_wer,
                delta,
            )
        else:
            logger.info("Sample %s WER: %.4f", sample.sample_id, wer)

        analytics.insert_eval_metric(
            sample_id=sample.sample_id,
            metric_name="wer",
            metric_value=wer,
            details=details,
        )
        evaluated += 1


def _resolve_gold_reference(sample_id: str, gold_transcripts: dict[str, str], transcript_path: Path | None) -> str | None:
    # Try direct sample ID match first, then normalized variants.
    candidates = [sample_id, sample_id.upper(), sample_id.replace("_", "-"), sample_id.replace("-", "_")]
    for candidate in candidates:
        if candidate in gold_transcripts:
            return gold_transcripts[candidate]

    # Fallback to transcript files if present.
    if transcript_path and transcript_path.exists():
        return Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
    return None
