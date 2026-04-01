import logging

from src.analytics.repository import AnalyticsRepository
from src.config.settings import get_settings
from src.ingestion.dataset_loader import DatasetLoader
from src.stt.mlx_whisper_service import MLXWhisperService

logger = logging.getLogger(__name__)


def run_stt_pipeline(limit: int | None = None) -> None:
    settings = get_settings()
    loader = DatasetLoader(
        recordings_dir=settings.recordings_dir,
        transcripts_dir=settings.transcripts_dir,
        casenotes_dir=settings.casenotes_dir,
    )
    samples = loader.load_samples()
    stt = MLXWhisperService()
    analytics = AnalyticsRepository()
    processed = 0
    skipped_invalid_audio = 0
    failed_other = 0
    for sample in samples:
        if limit is not None and processed >= limit:
            break
        if sample.recording_path is None:
            logger.warning("Skipping sample with no recording: %s", sample.sample_id)
            continue
        try:
            payload = stt.transcribe(sample.recording_path)
            analytics.insert_stt_run(sample_id=sample.sample_id, provider="mlx-whisper", payload=payload)
            processed += 1
        except Exception as exc:
            if stt.is_invalid_audio_error(exc):
                skipped_invalid_audio += 1
                logger.warning(
                    "Skipping invalid/corrupt audio for sample %s: %s",
                    sample.sample_id,
                    sample.recording_path,
                )
                continue
            failed_other += 1
            logger.exception("STT failed for sample %s", sample.sample_id)
            if limit is not None:
                # In sampled mode, continue scanning so limit still means successful transcriptions.
                continue
            raise
    logger.info(
        "STT pipeline finished. processed=%s skipped_invalid_audio=%s failed_other=%s",
        processed,
        skipped_invalid_audio,
        failed_other,
    )
