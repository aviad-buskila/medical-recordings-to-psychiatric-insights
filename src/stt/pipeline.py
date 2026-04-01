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
    total_candidates = sum(1 for s in samples if s.recording_path is not None)
    stt = MLXWhisperService()
    analytics = AnalyticsRepository()
    processed = 0
    skipped_invalid_audio = 0
    failed_other = 0
    seen_with_audio = 0
    logger.info(
        "Starting STT pipeline. candidates_with_audio=%s target_limit=%s",
        total_candidates,
        "all" if limit is None else limit,
    )
    for idx, sample in enumerate(samples, start=1):
        if limit is not None and processed >= limit:
            break
        if sample.recording_path is None:
            logger.warning("Skipping sample with no recording: %s", sample.sample_id)
            continue
        seen_with_audio += 1
        logger.info(
            "STT progress: sample=%s index=%s/%s successful=%s skipped_invalid=%s failed=%s",
            sample.sample_id,
            seen_with_audio,
            total_candidates,
            processed,
            skipped_invalid_audio,
            failed_other,
        )
        try:
            payload = stt.transcribe(sample.recording_path)
            analytics.insert_stt_run(sample_id=sample.sample_id, provider="mlx-whisper", payload=payload)
            processed += 1
            logger.info(
                "STT success: sample=%s successful=%s/%s",
                sample.sample_id,
                processed,
                "all" if limit is None else limit,
            )
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
