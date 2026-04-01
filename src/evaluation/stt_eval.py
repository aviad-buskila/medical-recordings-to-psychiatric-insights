import logging
from pathlib import Path

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.evaluation.wer import word_error_rate
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.pickle_loader import DatasetPickleLoader

logger = logging.getLogger(__name__)


def evaluate_stt_against_gold(limit: int | None = None, run_id: str | None = None) -> None:
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
    run_outputs = analytics.get_stt_outputs_for_run(run_id) if run_id else {}

    evaluated = 0
    for sample in samples:
        if limit is not None and evaluated >= limit:
            break
        reference = _resolve_gold_reference(sample.sample_id, gold_transcripts, sample.transcript_path)
        if not reference:
            continue
        hypothesis = run_outputs.get(sample.sample_id) if run_id else analytics.get_latest_stt_output(sample.sample_id)
        hypothesis = hypothesis or ""
        if not hypothesis:
            if run_id:
                logger.warning("No STT output found for sample_id=%s in run_id=%s", sample.sample_id, run_id)
            else:
                logger.warning("No STT output found in DB for sample_id=%s", sample.sample_id)
            continue
        wer = word_error_rate(reference=reference, hypothesis=hypothesis)
        analytics.insert_eval_metric(
            sample_id=sample.sample_id,
            metric_name="wer",
            metric_value=wer,
            details={"reference_source": "dataset.pickle", "run_id": run_id},
        )
        logger.info("Sample %s WER: %.4f", sample.sample_id, wer)
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
